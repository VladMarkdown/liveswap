"""
liveswap — Real-time face swap desktop application.
Entry point.
"""
from __future__ import annotations
import logging
import subprocess
import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QLabel, QPushButton,
    QTextEdit, QHBoxLayout,
)
from PySide6.QtCore import Qt, QThread, Signal

MODELS_DIR = Path("models")
# Present whenever download_models.py has fetched the live-mode models.
_REQUIRED_MODEL = MODELS_DIR / "GPEN-BFR-512.onnx"
_SOFTCAM_PATH = MODELS_DIR / "softcam.dll"


# ---------------------------------------------------------------------------
# Download dialog shown when models are missing
# ---------------------------------------------------------------------------

class _DownloadThread(QThread):
    line_ready = Signal(str)
    finished_ok = Signal(bool)

    def run(self):
        try:
            proc = subprocess.Popen(
                [sys.executable, "download_models.py"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in proc.stdout:
                self.line_ready.emit(line.rstrip())
            ok = proc.wait() == 0
            self.finished_ok.emit(ok)
        except Exception as e:
            self.line_ready.emit(f"Error: {e}")
            self.finished_ok.emit(False)


class _ModelsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("liveswap — Models Required")
        self.setMinimumSize(560, 360)
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        msg = QLabel(
            "Required models are not installed.\n\n"
            "Click 'Download Now' to download ~2 GB of model files.\n"
            "This only needs to happen once."
        )
        msg.setWordWrap(True)
        layout.addWidget(msg)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(self._log.font())
        self._log.hide()
        layout.addWidget(self._log)

        btns = QHBoxLayout()
        self._download_btn = QPushButton("Download Now")
        self._download_btn.setDefault(True)
        self._download_btn.clicked.connect(self._start_download)
        self._close_btn = QPushButton("Close")
        self._close_btn.clicked.connect(self.reject)
        btns.addWidget(self._download_btn)
        btns.addWidget(self._close_btn)
        layout.addLayout(btns)

        self._thread = None

    def _start_download(self):
        self._download_btn.setEnabled(False)
        self._log.show()
        self._thread = _DownloadThread(self)
        self._thread.line_ready.connect(lambda line: self._log.append(line))
        self._thread.finished_ok.connect(self._on_done)
        self._thread.start()

    def _on_done(self, ok: bool):
        if ok:
            self._log.append("\n✓ All models ready — restarting…")
            self.accept()
        else:
            self._log.append("\n✗ Some models failed. See output above.")
            self._download_btn.setEnabled(True)
            self._download_btn.setText("Retry")


def _softcam_present() -> bool:
    """Quick check: is the softcam DLL available without importing ctypes at module level."""
    import ctypes
    if _SOFTCAM_PATH.exists():
        return True
    if Path(r"C:\Windows\System32\softcam.dll").exists():
        return True
    try:
        ctypes.cdll.LoadLibrary("softcam")
        return True
    except OSError:
        return False


def _check_models(app: QApplication) -> bool:
    """Return True if all required models are present, else show download dialog."""
    if _REQUIRED_MODEL.exists():
        return True
    dlg = _ModelsDialog()
    result = dlg.exec()
    return result == QDialog.Accepted and _REQUIRED_MODEL.exists()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    # Without this the pipeline's INFO-level stage timings are never shown.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
    app = QApplication(sys.argv)
    app.setApplicationName("liveswap")
    app.setOrganizationName("liveswap")

    if not _check_models(app):
        sys.exit(0)

    from faceswap.gui.main_window import MainWindow
    window = MainWindow()
    window.show()

    # Non-blocking softcam warning (models are present but softcam not registered)
    if not _softcam_present():
        window.statusBar().showMessage(
            "Virtual camera not installed — run download_models.py as admin")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
