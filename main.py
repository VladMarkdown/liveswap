"""
liveswap — Real-time face swap desktop application.
Entry point.
"""
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from faceswap.gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("liveswap")
    app.setOrganizationName("liveswap")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
