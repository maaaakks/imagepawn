import sys

from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import MobileNet_V3_Large_Weights


"""
Can be used as a standalone script for predicting the FEN of a chessboard image.

This script takes a single input image of a chessboard and uses a deep learning model
(MobileNetV3 + custom prediction head) to extract the full FEN string.

The model expects the image to be centered and visually clean (224×224 or larger). It uses
standard ImageNet normalization and resizing transforms to ensure compatibility with pretrained backbones.

This script can be used standalone by running:
    python inference.py path_to_image.jpg

If no model is passed explicitly, it will load the checkpoint from 'models/best.pt' by default.

Example output:
    rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 1
"""

# Main args config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "models/best.pt" # DEFAULT PATH

def load_model(ckpt_path: Path) -> nn.Module:
    """Instantiate model architecture and load weights from checkpoint."""

    weights  = MobileNet_V3_Large_Weights.IMAGENET1K_V2
    backbone = models.mobilenet_v3_large(weights=weights).features
    model = nn.Sequential(backbone, BoardHead()).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

class BoardHead(nn.Module):
    def __init__(self, in_channels: int = 960):
        super().__init__()
        self.square_head   = nn.Conv2d(in_channels, 13, kernel_size=1)
        self.side_head     = nn.Linear(in_channels, 2)
        self.castling_head = nn.Linear(in_channels, 4)
        self.enpassant_head= nn.Linear(in_channels, 9)

    def forward(self, x):
        sq = self.square_head(x)                                            
        sq = nn.functional.interpolate(sq, (8, 8), mode="bilinear", align_corners=False)
        sq = sq.permute(0, 2, 3, 1)                                         

        pooled = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        side   = self.side_head(pooled)
        castle = self.castling_head(pooled)
        ep     = self.enpassant_head(pooled)
        return sq, side, castle, ep

idx2piece = {
    0: '',
    1: 'P',  2: 'N',  3: 'B',  4: 'R',  5: 'Q',  6: 'K',
    7: 'p',  8: 'n',  9: 'b', 10: 'r', 11: 'q', 12: 'k',
}
files = 'abcdefgh'

def squares_to_board_str(pred: torch.Tensor) -> str:
    """pred: 8×8 longs (0‑12) → FEN board substring."""
    rows = []
    for r in pred:
        empties = 0
        row_str = ''
        for v in r:
            v = int(v)
            if v == 0:
                empties += 1
            else:
                if empties:
                    row_str += str(empties); empties = 0
                row_str += idx2piece[v]
        if empties:
            row_str += str(empties)
        rows.append(row_str)
    return '/'.join(rows)


def decode_outputs(sq_logits, side_logits, castle_logits, ep_logits):
    # --- squares --------------------------------------------------------
    board_pred = sq_logits.argmax(-1)           # B×8×8 (long)
    board_str  = squares_to_board_str(board_pred[0])

    # --- side -----------------------------------------------------------
    side_idx   = side_logits.argmax(-1).item()  # 0 white, 1 black
    side_char  = 'w' if side_idx == 0 else 'b'

    # --- castling -------------------------------------------------------
    castle_prob= torch.sigmoid(castle_logits[0])  # 4‑vector
    flags = ''.join(f for f,p in zip('KQkq', castle_prob) if p > 0.5)
    castle_str = flags or '-'

    # --- en‑passant -----------------------------------------------------
    ep_idx = ep_logits[0].argmax().item()       # 0‑8
    if ep_idx == 0:
        ep_str = '-'
    else:
        file_ch = files[ep_idx - 1]
        rank_ch = '6' if side_char == 'b' else '3'  # side just played two‑square pawn move
        ep_str = f"{file_ch}{rank_ch}"

    # -------- final FEN (halfmove, fullmove default 0 1) ---------------
    return f"{board_str} {side_char} {castle_str} {ep_str} 0 1"


# === Main =====================================================================

def main(image, model=None):
    # Transforms 
    tfm = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    img = tfm(Image.open(image).convert("RGB")).unsqueeze(0).to(DEVICE)

    if model==None:
        model = load_model(model_path)

    with torch.no_grad():
        sq, side, castle, ep = model(img)
        fen = decode_outputs(sq, side, castle, ep)
    print(fen)

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    image = sys.argv[1] if len(sys.argv) > 1 else None
    main(image)
