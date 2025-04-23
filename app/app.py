#!/usr/bin/env python3
"""
Pawnalyzer
==========
Lightweight desktop utility that converts a screenshot of any chessboard
on your screen into a FEN string.

Features
--------

1. Screenshot – capture a chessboard from your screen and copied straight to your clipboard and displayed in the app interface.
2. Board preview – rendered as an SVG so you can verify the position visually.
3. One‑click Lichess link – opens the position in the Lichess board editor for analysis or sharing.


A tiny CNN (mobilenetv3) that can be run on any GPU in near-realtime, loaded from `models/best.pt`, 
detects every piece on the board and returning a complete FEN.

The app shows the FEN, copies it to your clipboard, displays the SVG board,
and provides a button to jump to Lichess and play with the position.

"""

from __future__ import annotations

import sys
import subprocess
import time
from pathlib import Path
from typing import Optional

import chess
import chess.svg
import torch
from PIL import Image, ImageQt, ImageGrab
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QUrl
from PyQt6.QtGui import QClipboard, QGuiApplication, QIcon, QDesktopServices
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QMessageBox
)

import inference

# Constants
CKPT_PATH = Path("models/best.pt")
ICON_PATH = Path("assets/pawnalyser.png")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SNIP_TIMEOUT = 8.0        # seconds
BOARD_SIZE = 350          # pixels

# Image transforms for the CNN
from torchvision import transforms as T

TFM = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


class LoaderThread(QThread):
    """Loads the neural network on a background thread."""

    finished = pyqtSignal(object)

    def run(self) -> None:
        model = None
        if CKPT_PATH.is_file():
            model = inference.load_model(CKPT_PATH)
        self.finished.emit(model)


launch_snipping_tool = lambda: subprocess.Popen(
    ["explorer", "ms-screenclip:"], close_fds=True)


class Pawnalyzer(QWidget):
    """Main application window."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.setWindowTitle("Pawnalyzer")
        if ICON_PATH.is_file():
            self.setWindowIcon(QIcon(str(ICON_PATH)))

        self.model = model

        self._poll_timer: Optional[QTimer] = None
        self._snip_start: float = 0.0

        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # Screenshot button
        top = QHBoxLayout()
        self.ss_btn = QPushButton("Screenshot")
        self.ss_btn.clicked.connect(self.start_screenshot)
        top.addWidget(self.ss_btn)
        top.addStretch()
        root.addLayout(top)

        # FEN field + actions
        fen_row = QHBoxLayout()
        self.fen_edit = QLineEdit(readOnly=True,
                                  placeholderText="FEN will appear here…")
        self.fen_edit.setMinimumWidth(500)
        fen_row.addWidget(self.fen_edit)

        self.copy_btn = QPushButton("Copy")
        self.copy_btn.clicked.connect(self.copy_fen)
        fen_row.addWidget(self.copy_btn)

        self.lichess_btn = QPushButton("Lichess")
        self.lichess_btn.clicked.connect(self.open_lichess)
        fen_row.addWidget(self.lichess_btn)

        fen_row.addStretch()
        root.addLayout(fen_row)

        # Board preview
        image_row = QHBoxLayout()
        self.svg_widget = QSvgWidget()
        self.svg_widget.setFixedSize(BOARD_SIZE, BOARD_SIZE)
        image_row.addWidget(self.svg_widget)
        root.addLayout(image_row)

    # Actions
    def open_lichess(self) -> None:
        fen = self.fen_edit.text()
        if not fen:
            QMessageBox.warning(self, "No FEN",
                                "There is no FEN to open in Lichess.")
            return
        QDesktopServices.openUrl(QUrl(f"https://lichess.org/editor/{fen}"))

    def start_screenshot(self) -> None:
        self.fen_edit.clear()
        self.svg_widget.load(b"")
        self.ss_btn.setEnabled(False)

        launch_snipping_tool()
        self._snip_start = time.monotonic()

        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._check_clipboard)
        self._poll_timer.start(300)

    def _check_clipboard(self) -> None:
        img = ImageGrab.grabclipboard()
        if isinstance(img, Image.Image):
            assert self._poll_timer
            self._poll_timer.stop()
            self._run_inference(img)
            return
        if time.monotonic() - self._snip_start > SNIP_TIMEOUT:
            assert self._poll_timer
            self._poll_timer.stop()
            self.ss_btn.setEnabled(True)

    def _run_inference(self, img: Image.Image) -> None:
        try:
            fen = self._infer(img)
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))
            self.ss_btn.setEnabled(True)
            return

        self.fen_edit.setText(fen)
        QGuiApplication.clipboard().setText(fen, QClipboard.Mode.Clipboard)

        svg = chess.svg.board(board=chess.Board(fen), size=BOARD_SIZE)
        self.svg_widget.load(svg.encode())

        self.ss_btn.setEnabled(True)

    def _infer(self, img: Image.Image) -> str:
        img = img.convert("RGB")
        tensor = TFM(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            sq, side, castle, ep = self.model(tensor)
            return inference.decode_outputs(sq, side, castle, ep)

    def copy_fen(self) -> None:
        QGuiApplication.clipboard().setText(
            self.fen_edit.text(), QClipboard.Mode.Clipboard)


def main() -> None:
    app = QApplication(sys.argv)
    if ICON_PATH.is_file():
        app.setWindowIcon(QIcon(str(ICON_PATH)))

    splash = QLabel("Loading model… please wait",
                    alignment=Qt.AlignmentFlag.AlignCenter)
    splash.setWindowFlag(Qt.WindowType.FramelessWindowHint)
    splash.setFixedSize(220, 90)
    splash.show()

    loader = LoaderThread()
    loader.finished.connect(lambda m: _on_loaded(m, app, splash))
    loader.start()

    sys.exit(app.exec())


def _on_loaded(model: object, app: QApplication, splash: QLabel) -> None:
    splash.close()
    if model is None:
        QMessageBox.critical(None, "Checkpoint missing",
                             f"Cannot find {CKPT_PATH}")
        sys.exit(1)
    app.main_window = Pawnalyzer(model)
    app.main_window.show()


if __name__ == "__main__":
    main()
