import random, sys, os

from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import MobileNet_V3_Large_Weights

# Main args config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "dataset"
EPOCHS = 30
BATCH_SIZE = 32
lr = 1e-3
WORKERS = 4
FREEZE = False

class RandomArrowOverlay:
    """
    Data-augmentation:
    Overlay a semi‑transparent arrow (or, with 30 % probability, a square highlight) with probability *p*.
    Very important part considering there is a lot of arrow noise in the real data.
    This noise can significally reduce the performance of the model.
    """
    def __init__(self, p: float = 0.4):
        self.p = p
        # colours inspired by lichess / chess.com (RGBA)
        self.arrow_cols = [
            (255, 50, 50, 160),   # red
            (0, 120, 215, 160),   # blue
            (0, 200, 70, 160),    # green
        ]
        self.square_cols = [
            (0, 120, 215,  90),   # light blue (highlight)
            (255,  60, 60,  90),  # light red
            (50, 200, 50,  90),   # light green
        ]

    # board square centre → pixel coords in 224×224 crop
    def _c(self, f: float, r: float) -> Tuple[int, int]:
        return int((f + 0.5) * 224 / 8), int((7 - r + 0.5) * 224 / 8)

    def _arrow(self, draw: ImageDraw.ImageDraw):
        a, b = random.sample(range(64), 2)
        fx, fy, tx, ty = a % 8, a // 8, b % 8, b // 8
        start, end = self._c(fx, fy), self._c(tx, ty)
        col = random.choice(self.arrow_cols)
        w = random.randint(4, 6)
        draw.line([start, end], fill=col, width=w)
        # arrow head (simple triangle)
        vx, vy = end[0] - start[0], end[1] - start[1]
        L = max((vx * vx + vy * vy) ** 0.5, 1)
        ux, uy = vx / L, vy / L
        left  = (end[0] - 8 * ux + 4 * uy, end[1] - 8 * uy - 4 * ux)
        right = (end[0] - 8 * ux - 4 * uy, end[1] - 8 * uy + 4 * ux)
        draw.polygon([end, left, right], fill=col)

    def _square(self, draw: ImageDraw.ImageDraw):
        s = random.randrange(64)
        f, r = s % 8, s // 8
        x0, y0 = self._c(f - 0.5, r + 0.5)
        x1, y1 = self._c(f + 0.5, r - 0.5)
        draw.rectangle([x0, y0, x1, y1], fill=random.choice(self.square_cols))

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        img = img.convert("RGBA")
        draw = ImageDraw.Draw(img, "RGBA")
        if random.random() < 0.7:
            self._arrow(draw)
        else:
            self._square(draw)
        return img.convert("RGB")


class ChessFENBoardDataset(Dataset):
    """
    Dataset for training the model.
    The images are cropped to a 224×224 square, and then the board is encoded into a tensor of shape (8, 8),
    where each square is mapped to an integer representing a specific piece type (or empty square).
    """

    _piece_map = {
        "P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
        "p": 7, "n": 8, "b": 9, "r": 10, "q": 11, "k": 12,
    }

    def __init__(self, files: List[Path], transform):
        self.files = files
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def _fen_to_labels(self, fen: str):
        """Split the filename‑encoded FEN into tensors suitable for training."""
        parts = fen.split("_")
        if len(parts) == 1: 
            fen += "_w_-_-_0_1"; parts = fen.split("_")
        board_str, side_char, castles, ep_sq = parts[0], parts[1], parts[2], parts[3]

        rows = board_str.replace("-", "/").split("/")
        board = torch.zeros((8, 8), dtype=torch.long)
        for r, row in enumerate(rows):
            c = 0
            for ch in row:
                if ch.isdigit():
                    c += int(ch)
                else:
                    board[r, c] = self._piece_map[ch]
                    c += 1

        side = torch.tensor(0 if side_char == "w" else 1, dtype=torch.long)
        castling = torch.tensor([1.0 if f in castles else 0.0 for f in "KQkq"], dtype=torch.float32)
        ep = torch.tensor(0 if ep_sq == "-" else (ord(ep_sq[0]) - ord("a") + 1), dtype=torch.long)
        return board, side, castling, ep

    #  __getitem__ 
    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        board, side, castling, ep = self._fen_to_labels(path.stem)
        if self.transform:
            img = self.transform(img)
        return img, board, side, castling, ep
    
class BoardHead(nn.Module):
    """
    This module predicts everything needed to rebuild a full FEN from an image.

    It includes separate heads to estimate:
    - The board layout (which pieces are on which squares),
    - Whose turn it is (white or black),
    - The available castling rights,
    - And whether an en passant move is possible.

    These elements go beyond just recognizing pieces — they capture the current game state,
    which is essential for understanding how the position arose and what moves are legal.
    """
    def __init__(self, in_channels: int = 960):
        super().__init__()
        self.square_head   = nn.Conv2d(in_channels, 13, 1)
        self.side_head     = nn.Linear(in_channels, 2)
        self.castling_head = nn.Linear(in_channels, 4)
        self.ep_head       = nn.Linear(in_channels, 9)

    def forward(self, x):
        sq_logits = nn.functional.interpolate(self.square_head(x), (8, 8), mode="bilinear", align_corners=False)
        sq_logits = sq_logits.permute(0, 2, 3, 1)  # B×8×8×13
        pooled = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        side_logits     = self.side_head(pooled)
        castle_logits   = self.castling_head(pooled)
        ep_logits       = self.ep_head(pooled)
        return sq_logits, side_logits, castle_logits, ep_logits


# Metrics
@torch.no_grad()
def compute_metrics(pred: Dict[str, torch.Tensor], tgt: Dict[str, torch.Tensor]) -> Dict[str, float]:
    sq_pred = pred["sq"].argmax(-1)
    sq_tgt  = tgt["board"]
    per_sq_acc = (sq_pred == sq_tgt).float().mean().item()
    full_board_acc = (sq_pred == sq_tgt).view(sq_pred.size(0), -1).all(dim=1).float().mean().item()
    side_acc = (pred["side"].argmax(-1) == tgt["side"]).float().mean().item()
    castle_pred = (torch.sigmoid(pred["castle"]) > 0.5).float()
    castle_acc = (castle_pred == tgt["castling"]).all(dim=1).float().mean().item()
    ep_acc = (pred["ep"].argmax(-1) == tgt["ep"]).float().mean().item()
    return {
        "per_sq_acc": per_sq_acc,
        "full_board_acc": full_board_acc,
        "side_acc": side_acc,
        "castle_acc": castle_acc,
        "ep_acc": ep_acc,
    }

# Loss
criterion = {
    "square":    nn.CrossEntropyLoss(),
    "side":      nn.CrossEntropyLoss(),
    "castling":  nn.BCEWithLogitsLoss(),
    "ep":        nn.CrossEntropyLoss(),
}

def loss_fn(out: Dict[str, torch.Tensor], tgt: Dict[str, torch.Tensor]):
    l_sq   = criterion["square"](out["sq"].reshape(-1, 13), tgt["board"].reshape(-1))
    l_side = criterion["side"](out["side"], tgt["side"])
    l_cast = criterion["castling"](out["castle"], tgt["castling"])
    l_ep   = criterion["ep"](out["ep"], tgt["ep"])
    # heavier weight on side‑to‑move because a flipped colour ruins the full‑board FEN
    return 0.5 * l_sq + 2.0 * l_side + l_cast + l_ep

def forward_dict(model: nn.Module, x):
    sq, side, castle, ep = model(x)
    return {"sq": sq, "side": side, "castle": castle, "ep": ep}

# Train
def run_epoch(model: nn.Module, loader: DataLoader, optim: torch.optim.Optimizer | None = None):
    train_mode = optim is not None
    model.train() if train_mode else model.eval()
    agg_loss, metric_sum, batches = 0.0, {k: 0.0 for k in [
        "per_sq_acc", "full_board_acc", "side_acc", "castle_acc", "ep_acc"]}, 0

    pbar = tqdm(loader, leave=False)
    for imgs, board, side, castle, ep in pbar:
        imgs, board, side, castle, ep = (t.to(DEVICE, non_blocking=True) for t in (imgs, board, side, castle, ep))
        tgt = {"board": board, "side": side, "castling": castle, "ep": ep}
        out = forward_dict(model, imgs)
        loss = loss_fn(out, tgt)

        if train_mode:
            optim.zero_grad(); loss.backward(); optim.step()

        metrics = compute_metrics(out, tgt)
        for k, v in metrics.items():
            metric_sum[k] += v
        agg_loss += loss.item(); batches += 1
        pbar.set_postfix(loss=agg_loss / batches, sqAcc=metric_sum["per_sq_acc"] / batches)

    # average metrics
    return {"loss": agg_loss / batches, **{k: v / batches for k, v in metric_sum.items()}}

# Main
def main(resume=None):
    # Transforms
    stats = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.RandomRotation(6),
        RandomArrowOverlay(0.4),
        transforms.ColorJitter(0.25, 0.25, 0.25, 0.08),
        transforms.ToTensor(),
        transforms.Normalize(**stats),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(**stats)
    ])

    def collect(p: Path):
        return [f for patt in ("*.png", "*.jpg", "*.jpeg") for f in p.rglob(patt) if f.is_file()]

    train_files = collect(Path(DATA_DIR) / "train")
    val_files   = collect(Path(DATA_DIR) / "val")
    if not val_files:
        random.shuffle(train_files)
        split = int(0.9 * len(train_files))
        train_files, val_files = train_files[:split], train_files[split:]

    train_ds = ChessFENBoardDataset(train_files, train_tf)
    val_ds   = ChessFENBoardDataset(val_files,   val_tf)
    train_dl = DataLoader(train_ds, BATCH_SIZE=BATCH_SIZE, shuffle=True,
                          num_WORKERS=WORKERS, pin_memory=True)
    val_dl   = DataLoader(val_ds, BATCH_SIZE=BATCH_SIZE, shuffle=False,
                          num_WORKERS=WORKERS, pin_memory=True)

    """
    OPTIONAL: freeze lower backbone layers
    This freeze the early layers of the MobileNetV3 backbone (typically the first ~6 of 16 blocks),
    which are responsible for extracting low-level features like edges and textures.
    Freezing them can speed up training and reduce overfitting, especially if using pretrained weights.
    These layers are less task-specific, so retraining them is often unnecessary for fine-tuning.
    """
    backbone = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2).features
    if FREEZE:
        for p in backbone[:6].parameters():
            p.requires_grad = False
    model = nn.Sequential(backbone, BoardHead()).to(DEVICE)

    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=len(train_dl) * EPOCHS)

    start_epoch, best_board_acc = 1, 0.0

    # Optional resume from checkpoint
    if resume is not None and os.path.isfile(resume):
        ckpt = torch.load(resume, map_location=DEVICE)
        model.load_state_dict(ckpt["model"], strict=False)

        if "optim" in ckpt and ckpt["optim"]:
            try:
                optim.load_state_dict(ckpt["optim"])
            except ValueError:
                print("[warning] Optimizer mismatch – starting optimizer fresh.")

        if "sched" in ckpt and ckpt["sched"]:
            try:
                sched.load_state_dict(ckpt["sched"])
            except ValueError:
                print("[warning] Scheduler mismatch – starting scheduler fresh.")

        start_epoch = ckpt.get("epoch", 0) + 1
        best_board_acc = ckpt.get("metrics", {}).get("full_board_acc", 0.0)

        print(f"→ Resumed from '{resume}' @ epoch {start_epoch - 1} (fullBoardAcc={best_board_acc:.3f})")

    elif resume is not None:
        print(f"[error] Checkpoint '{resume}' not found", file=sys.stderr)
        sys.exit(1)


    # Training loop 
    ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)
    for epoch in range(start_epoch, EPOCHS + 1):
        train_res = run_epoch(model, train_dl, optim)
        sched.step()
        val_res = run_epoch(model, val_dl)
        print(f"Epoch {epoch:03d} | train loss {train_res['loss']:.4f} | val loss {val_res['loss']:.4f} | "
              f"fullBoard {val_res['full_board_acc']*100:.2f}%")

        ckpt = {
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "sched": sched.state_dict(),
            "epoch": epoch,
            "metrics": val_res,
        }
        torch.save(ckpt, ckpt_dir / f"ckpt_{epoch:03d}.pt")
        if val_res["full_board_acc"] > best_board_acc:
            best_board_acc = val_res["full_board_acc"]
            torch.save(ckpt, ckpt_dir / "best.pt")
            
if __name__ == "__main__":
    resume = sys.argv[1] if len(sys.argv) > 1 else None
    main(resume)
