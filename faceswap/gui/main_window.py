"""
Main application window (PySide6) — simplified UX for non-technical users.

No tabs at startup. Welcome screen → choose mode → work.
All long-running ops run in background threads via WorkerSignals.
Settings persisted to ~/.liveswap/settings.json.

Internal attribute names (required by test_gui_lifecycle.py):
  _tabs, _profile_combo, _start_btn, _quality_ultra, _cam_combo,
  _runner, _cap, _pipeline, _vcam, _signals, _starting, _closing,
  _source_face, _swapper, _enhancer, _parser
"""
from __future__ import annotations

import gc
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal, QObject, Slot, QSize
from PySide6.QtGui import QImage, QPixmap, QFont, QPainter, QColor
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFileDialog, QProgressBar, QCheckBox, QLineEdit,
    QStatusBar, QSizePolicy, QStackedWidget, QScrollArea, QFrame,
    QDialog, QFormLayout, QSlider, QGroupBox, QToolButton,
)

# ---------------------------------------------------------------------------
# Settings helpers
# ---------------------------------------------------------------------------

_SETTINGS_PATH = Path.home() / ".liveswap" / "settings.json"
_DEFAULT_SETTINGS: dict = {
    "camera_index": 0,
    "quality": "ultra",
    "vcam_enabled": True,
    "pixel_boost": False,
    "pct_harmonize": False,
    "occlusion": False,
    "webrtc_enabled": False,
    "last_photo": "",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# ---------------------------------------------------------------------------
# Module-level lazy imports — referenced as globals by AST-tested methods.
# The test namespace overrides these; production gets the real functions.
# ---------------------------------------------------------------------------
try:
    from faceswap.profile.manager import load_profile_by_name
    from faceswap.profile.synthesizer import generate_profile_source_face
    from faceswap.core.pipeline import SwapPipeline
except Exception:
    load_profile_by_name = None       # type: ignore[assignment]
    generate_profile_source_face = None  # type: ignore[assignment]
    SwapPipeline = None               # type: ignore[assignment]


def _load_settings() -> dict:
    try:
        with open(_SETTINGS_PATH) as f:
            loaded = json.load(f)
        s = dict(_DEFAULT_SETTINGS)
        s.update(loaded)
        return s
    except Exception:
        return dict(_DEFAULT_SETTINGS)


def _save_settings(s: dict) -> None:
    _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_SETTINGS_PATH, "w") as f:
        json.dump(s, f, indent=2)


# ---------------------------------------------------------------------------
# Dark theme stylesheet
# ---------------------------------------------------------------------------

_STYLE = """
QMainWindow, QWidget { background: #1a1a1a; color: #f0f0f0; }
QPushButton {
    background: #2d2d2d; color: #f0f0f0; border: 1px solid #444;
    border-radius: 8px; padding: 8px 16px; font-size: 14px;
}
QPushButton:hover { background: #3a3a3a; }
QPushButton:pressed { background: #252525; }
QPushButton#start_btn {
    background: #2e7d32; color: #fff; font-size: 18px; font-weight: bold;
    border: none; border-radius: 12px; padding: 16px 40px;
}
QPushButton#start_btn:hover { background: #388e3c; }
QPushButton#stop_btn {
    background: #c62828; color: #fff; font-size: 14px;
    border: none; border-radius: 8px; padding: 8px 20px;
}
QPushButton#card_btn {
    background: #232323; border: 2px solid #333; border-radius: 16px;
    padding: 32px 24px; font-size: 16px; text-align: center;
}
QPushButton#card_btn:hover { background: #2a2a2a; border-color: #4CAF50; }
QComboBox {
    background: #2d2d2d; color: #f0f0f0; border: 1px solid #444;
    border-radius: 6px; padding: 6px 12px; font-size: 13px;
}
QComboBox QAbstractItemView { background: #2d2d2d; color: #f0f0f0; }
QLineEdit {
    background: #2d2d2d; color: #f0f0f0; border: 1px solid #444;
    border-radius: 6px; padding: 8px 12px; font-size: 14px;
}
QProgressBar {
    background: #2d2d2d; border: 1px solid #444; border-radius: 6px;
    text-align: center; color: #f0f0f0;
}
QProgressBar::chunk { background: #4CAF50; border-radius: 5px; }
QLabel#section_title { font-size: 15px; font-weight: bold; color: #aaa; }
QLabel#big_title { font-size: 32px; font-weight: bold; color: #fff; }
QLabel#subtitle { font-size: 15px; color: #888; }
QCheckBox { font-size: 13px; }
QCheckBox::indicator { width: 20px; height: 20px; border-radius: 4px;
                        border: 1px solid #555; background: #2d2d2d; }
QCheckBox::indicator:checked { background: #4CAF50; border-color: #4CAF50; }
QScrollArea { border: none; }
QFrame#drop_zone {
    border: 2px dashed #555; border-radius: 12px; background: #232323;
}
QFrame#drop_zone:hover { border-color: #4CAF50; }
QFrame#profile_card {
    background: #2a2a2a; border: 1px solid #3a3a3a; border-radius: 10px;
}
"""

# ---------------------------------------------------------------------------
# Worker signals
# ---------------------------------------------------------------------------

class WorkerSignals(QObject):
    startup_ready = Signal(object)   # (pipeline, cap) tuple
    failed        = Signal(str, str) # (title, message)
    source_ready  = Signal(object, str, float)  # (face, filename, confidence_pct)
    profile_done  = Signal(str, float)           # (name, quality_score)
    progress_msg  = Signal(str)


# ---------------------------------------------------------------------------
# Internal mode tracker (satisfies _tabs.currentIndex() contract in tests)
# ---------------------------------------------------------------------------

class _ModeIndex:
    """Minimal object that satisfies `_tabs.currentIndex()` checks in tests."""
    LIVE    = 0
    PROFILE = 1

    def __init__(self):
        self._idx = self.LIVE

    def currentIndex(self) -> int:
        return self._idx

    def setCurrentIndex(self, idx: int) -> None:
        self._idx = idx


# ---------------------------------------------------------------------------
# Drop zone widget
# ---------------------------------------------------------------------------

class _DropZone(QFrame):
    files_dropped = Signal(list)

    def __init__(self, text: str, multiple: bool = False, parent=None):
        super().__init__(parent)
        self.setObjectName("drop_zone")
        self._multiple = multiple
        self.setAcceptDrops(True)
        self.setMinimumHeight(140)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        self._icon = QLabel("📁", self)
        self._icon.setAlignment(Qt.AlignCenter)
        self._icon.setStyleSheet("font-size: 40px; border: none;")
        self._label = QLabel(text, self)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setWordWrap(True)
        self._label.setStyleSheet("font-size: 14px; color: #aaa; border: none;")
        layout.addWidget(self._icon)
        layout.addWidget(self._label)
        self.setCursor(Qt.PointingHandCursor)

    def set_text(self, text: str) -> None:
        self._label.setText(text)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self._multiple:
                paths, _ = QFileDialog.getOpenFileNames(
                    self, "Choose photos", "",
                    "Images (*.jpg *.jpeg *.png *.bmp *.webp)")
            else:
                path, _ = QFileDialog.getOpenFileName(
                    self, "Choose photo", "",
                    "Images (*.jpg *.jpeg *.png *.bmp *.webp)")
                paths = [path] if path else []
            if paths:
                self.files_dropped.emit(paths)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls()
                 if Path(u.toLocalFile()).suffix.lower() in IMG_EXTS]
        if paths:
            self.files_dropped.emit(paths)


# ---------------------------------------------------------------------------
# Photo thumbnail strip
# ---------------------------------------------------------------------------

class _PhotoStrip(QWidget):
    def __init__(self, max_photos: int = 10, parent=None):
        super().__init__(parent)
        self._max = max_photos
        self._paths: list[str] = []
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self._cells: list[QLabel] = []
        for i in range(max_photos):
            cell = QLabel()
            cell.setFixedSize(60, 60)
            cell.setAlignment(Qt.AlignCenter)
            cell.setStyleSheet("background:#232323; border-radius:6px; border:1px solid #333;")
            layout.addWidget(cell)
            self._cells.append(cell)
        layout.addStretch()

    def set_paths(self, paths: list[str]) -> None:
        self._paths = paths[:self._max]
        for i, cell in enumerate(self._cells):
            if i < len(self._paths):
                img = cv2.imread(self._paths[i])
                if img is not None:
                    img = cv2.resize(img, (58, 58))
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    qimg = QImage(rgb.data, 58, 58, 58 * 3, QImage.Format_RGB888)
                    cell.setPixmap(QPixmap.fromImage(qimg))
                else:
                    cell.setText("?")
            else:
                cell.setPixmap(QPixmap())
                cell.setText("")


# ---------------------------------------------------------------------------
# Settings dialog
# ---------------------------------------------------------------------------

class _SettingsDialog(QDialog):
    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(380)
        self._s = dict(settings)
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Virtual camera
        self._vcam_cb = QCheckBox("Virtual camera (DirectShow Softcam)")
        self._vcam_cb.setChecked(self._s.get("vcam_enabled", True))
        layout.addWidget(self._vcam_cb)

        # Quality
        qlayout = QHBoxLayout()
        qlayout.addWidget(QLabel("Quality:"))
        self._quality_fast = QPushButton("Fast")
        self._quality_fast.setCheckable(True)
        self._quality_ultra = QPushButton("Ultra")
        self._quality_ultra.setCheckable(True)
        q = self._s.get("quality", "ultra")
        self._quality_fast.setChecked(q == "fast")
        self._quality_ultra.setChecked(q == "ultra")
        self._quality_fast.clicked.connect(lambda: self._set_quality("fast"))
        self._quality_ultra.clicked.connect(lambda: self._set_quality("ultra"))
        qlayout.addWidget(self._quality_fast)
        qlayout.addWidget(self._quality_ultra)
        qlayout.addStretch()
        layout.addLayout(qlayout)

        # Advanced section
        adv_btn = QPushButton("▼ Advanced")
        adv_btn.setFlat(True)
        adv_btn.setStyleSheet("color: #888; font-size: 12px; text-align: left; padding: 4px;")
        layout.addWidget(adv_btn)
        self._adv_box = QGroupBox()
        self._adv_box.hide()
        adv_inner = QVBoxLayout(self._adv_box)
        self._pixel_boost_cb = QCheckBox("Pixel boost (4× patch tiling, slower)")
        self._pixel_boost_cb.setChecked(self._s.get("pixel_boost", False))
        self._pct_cb = QCheckBox("PCT color harmonization (better lighting match)")
        self._pct_cb.setChecked(self._s.get("pct_harmonize", False))
        self._occ_cb = QCheckBox("Occlusion detection (hands/objects)")
        self._occ_cb.setChecked(self._s.get("occlusion", False))
        self._webrtc_cb = QCheckBox("Stream to phone (WebRTC, port 8765)")
        self._webrtc_cb.setChecked(self._s.get("webrtc_enabled", False))
        for w in (self._pixel_boost_cb, self._pct_cb, self._occ_cb, self._webrtc_cb):
            adv_inner.addWidget(w)
        layout.addWidget(self._adv_box)
        adv_btn.clicked.connect(lambda: (
            self._adv_box.show() if self._adv_box.isHidden() else self._adv_box.hide()))

        btns = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept)
        btns.addStretch()
        btns.addWidget(ok_btn)
        layout.addLayout(btns)

    def _set_quality(self, q: str) -> None:
        self._quality_fast.setChecked(q == "fast")
        self._quality_ultra.setChecked(q == "ultra")

    def result_settings(self) -> dict:
        q = "fast" if self._quality_fast.isChecked() else "ultra"
        return dict(self._s,
                    vcam_enabled=self._vcam_cb.isChecked(),
                    quality=q,
                    pixel_boost=self._pixel_boost_cb.isChecked(),
                    pct_harmonize=self._pct_cb.isChecked(),
                    occlusion=self._occ_cb.isChecked(),
                    webrtc_enabled=self._webrtc_cb.isChecked())


# ---------------------------------------------------------------------------
# Welcome page
# ---------------------------------------------------------------------------

class _WelcomePage(QWidget):
    quick_start = Signal()
    high_quality = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setAlignment(Qt.AlignCenter)
        outer.setSpacing(24)

        title = QLabel("liveswap")
        title.setObjectName("big_title")
        title.setAlignment(Qt.AlignCenter)
        outer.addWidget(title)

        sub = QLabel("Real-time face swap")
        sub.setObjectName("subtitle")
        sub.setAlignment(Qt.AlignCenter)
        outer.addWidget(sub)

        cards = QHBoxLayout()
        cards.setSpacing(24)

        quick = QPushButton("⚡ Quick Start\n\n1 photo, instant\n\nDrop a photo, pick your camera\nand go live immediately")
        quick.setObjectName("card_btn")
        quick.setMinimumSize(240, 200)
        quick.setWordWrap(True)
        quick.clicked.connect(self.quick_start)
        cards.addWidget(quick)

        hq = QPushButton("⭐ High Quality\n\nMultiple photos, 2 min setup\n\nBetter identity preservation\nwith a named profile")
        hq.setObjectName("card_btn")
        hq.setMinimumSize(240, 200)
        hq.setWordWrap(True)
        hq.clicked.connect(self.high_quality)
        cards.addWidget(hq)

        outer.addLayout(cards)
        outer.addStretch()


# ---------------------------------------------------------------------------
# Live mode page
# ---------------------------------------------------------------------------

class _LivePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(32, 24, 32, 24)

        self._drop = _DropZone("👤  Drop your photo here\nor click to browse", multiple=False)
        self._drop.files_dropped.connect(self._on_photo)
        layout.addWidget(self._drop)

        self._photo_label = QLabel()
        self._photo_label.setAlignment(Qt.AlignCenter)
        self._photo_label.hide()
        layout.addWidget(self._photo_label)

        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("Camera:"))
        self.cam_combo = QComboBox()
        self._populate_cameras()
        cam_row.addWidget(self.cam_combo)
        cam_row.addStretch()
        layout.addLayout(cam_row)

        self.start_btn = QPushButton("▶  Start")
        self.start_btn.setObjectName("start_btn")
        self.start_btn.setEnabled(False)
        layout.addWidget(self.start_btn, alignment=Qt.AlignCenter)

        self._hint = QLabel("💡 After starting: open your video app → select 'DirectShow Softcam'")
        self._hint.setAlignment(Qt.AlignCenter)
        self._hint.setStyleSheet("font-size: 12px; color: #666;")
        layout.addWidget(self._hint)

        layout.addStretch()
        self._photo_path: str = ""

    def _populate_cameras(self) -> None:
        self.cam_combo.clear()
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                self.cam_combo.addItem(f"Camera {i}", userData=i)
                cap.release()
        if self.cam_combo.count() == 0:
            self.cam_combo.addItem("No camera found", userData=0)

    def _on_photo(self, paths: list[str]) -> None:
        self._photo_path = paths[0]
        self.start_btn.setEnabled(True)
        # Show thumbnail
        img = cv2.imread(self._photo_path)
        if img is not None:
            h, w = img.shape[:2]
            scale = 100 / max(h, w, 1)
            small = cv2.resize(img, (int(w * scale), int(h * scale)))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format_RGB888)
            self._photo_label.setPixmap(QPixmap.fromImage(qimg).copy())
            self._photo_label.show()
            self._drop.set_text(f"✓ {Path(self._photo_path).name}")

    @property
    def photo_path(self) -> str:
        return self._photo_path

    @property
    def camera_index(self) -> int:
        d = self.cam_combo.currentData()
        return d if d is not None else 0


# ---------------------------------------------------------------------------
# Profile mode page
# ---------------------------------------------------------------------------

class _ProfilePage(QWidget):
    create_requested = Signal(list, str)   # paths, name

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(32, 24, 32, 24)

        # Saved profiles scroll area
        self._profiles_label = QLabel("Saved profiles:")
        self._profiles_label.setObjectName("section_title")
        layout.addWidget(self._profiles_label)

        self._profiles_scroll = QScrollArea()
        self._profiles_scroll.setWidgetResizable(True)
        self._profiles_container = QWidget()
        self._profiles_layout = QHBoxLayout(self._profiles_container)
        self._profiles_layout.setSpacing(12)
        self._profiles_layout.addStretch()
        self._profiles_scroll.setWidget(self._profiles_container)
        self._profiles_scroll.setMaximumHeight(130)
        layout.addWidget(self._profiles_scroll)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #333;")
        layout.addWidget(sep)

        # Create new profile area
        layout.addWidget(QLabel("Create new profile:"))

        self._drop = _DropZone(
            "📸  Drop 5–10 photos of the person\nor click to browse",
            multiple=True,
        )
        self._drop.files_dropped.connect(self._on_photos)
        layout.addWidget(self._drop)

        self._strip = _PhotoStrip(max_photos=10)
        layout.addWidget(self._strip)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. John")
        name_row.addWidget(self._name_edit)
        layout.addLayout(name_row)

        self.create_btn = QPushButton("⭐  Create Profile")
        self.create_btn.setObjectName("start_btn")
        self.create_btn.setEnabled(False)
        self.create_btn.clicked.connect(self._on_create)
        layout.addWidget(self.create_btn, alignment=Qt.AlignCenter)

        self._progress = QProgressBar()
        self._progress.hide()
        self._progress_label = QLabel()
        self._progress_label.setAlignment(Qt.AlignCenter)
        self._progress_label.hide()
        layout.addWidget(self._progress)
        layout.addWidget(self._progress_label)

        layout.addStretch()

        self._photo_paths: list[str] = []

        # Hidden profile selector for internal _start_pipeline logic
        self.profile_combo = QComboBox()
        self.profile_combo.hide()
        layout.addWidget(self.profile_combo)

        self.refresh_profiles()

    def _on_photos(self, paths: list[str]) -> None:
        self._photo_paths = paths[:10]
        self._strip.set_paths(self._photo_paths)
        self._drop.set_text(f"✓ {len(self._photo_paths)} / 10 photos")
        self.create_btn.setEnabled(bool(self._name_edit.text().strip()))

    def _on_create(self) -> None:
        name = self._name_edit.text().strip()
        if not name or not self._photo_paths:
            return
        self.create_btn.setEnabled(False)
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.show()
        self._progress_label.setText("Analyzing photos…")
        self._progress_label.show()
        self.create_requested.emit(self._photo_paths, name)

    def show_progress(self, msg: str) -> None:
        self._progress_label.setText(msg)

    def show_done(self, name: str, quality: float) -> None:
        self._progress.hide()
        self._progress_label.setText(
            f"✓ Profile '{name}' created — identity strength: {int(quality * 100)}%")
        self._name_edit.clear()
        self._photo_paths = []
        self._strip.set_paths([])
        self._drop.set_text("📸  Drop 5–10 photos of the person\nor click to browse")
        self.create_btn.setEnabled(False)
        self.refresh_profiles()

    def refresh_profiles(self) -> None:
        try:
            from faceswap.profile.manager import list_profiles
            profiles = list_profiles()
        except Exception:
            profiles = []

        # Update hidden combo
        self.profile_combo.clear()
        if not profiles:
            self.profile_combo.addItem("(no profiles)")
        else:
            for name in profiles:
                self.profile_combo.addItem(name)

        # Update profile cards
        for i in reversed(range(self._profiles_layout.count() - 1)):
            item = self._profiles_layout.itemAt(i)
            if item and item.widget():
                item.widget().deleteLater()

        if not profiles:
            lbl = QLabel("No profiles yet")
            lbl.setStyleSheet("color: #666; font-size: 13px;")
            self._profiles_layout.insertWidget(0, lbl)
        else:
            for name in profiles:
                card = QPushButton(f"👤 {name}")
                card.setObjectName("card_btn")
                card.setFixedHeight(90)
                card.setMinimumWidth(120)
                # Select this profile when clicked
                card.clicked.connect(lambda checked=False, n=name: self._select_profile(n))
                self._profiles_layout.insertWidget(0, card)

    def _select_profile(self, name: str) -> None:
        # Sync hidden combo so _start_pipeline picks up the right profile
        idx = self.profile_combo.findText(name)
        if idx >= 0:
            self.profile_combo.setCurrentIndex(idx)


# ---------------------------------------------------------------------------
# Preview overlay (shown when pipeline is running)
# ---------------------------------------------------------------------------

class _PreviewOverlay(QWidget):
    stop_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._video = QLabel()
        self._video.setAlignment(Qt.AlignCenter)
        self._video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._video, stretch=1)

        self._stats = QLabel()
        self._stats.setAlignment(Qt.AlignCenter)
        self._stats.setStyleSheet("font-size: 11px; color: #888;")
        layout.addWidget(self._stats)

        stop_btn = QPushButton("■  Stop")
        stop_btn.setObjectName("stop_btn")
        stop_btn.clicked.connect(self.stop_requested)
        layout.addWidget(stop_btn, alignment=Qt.AlignCenter)

    def show_frame(self, bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        self._video.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self._video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def set_stats(self, text: str) -> None:
        self._stats.setText(text)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("liveswap")
        self.setMinimumSize(900, 640)
        self.setStyleSheet(_STYLE)

        self._settings = _load_settings()

        # --- Worker signals ---
        self._signals = WorkerSignals()
        self._signals.startup_ready.connect(self._on_startup_ready)
        self._signals.failed.connect(self._on_error)
        self._signals.source_ready.connect(self._on_source_ready)
        self._signals.profile_done.connect(self._on_profile_done)
        self._signals.progress_msg.connect(self._on_progress_msg)

        # --- Internal state (required by tests) ---
        self._runner   = None
        self._cap      = None
        self._pipeline = None
        self._vcam     = None
        self._swapper  = None
        self._enhancer = None
        self._parser   = None
        self._source_face = None
        self._starting = False
        self._closing  = False
        self._running  = False

        # --- Mode tracker (satisfies _tabs.currentIndex() contract) ---
        self._tabs = _ModeIndex()

        # --- Build UI ---
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Top navigation bar (hidden on welcome)
        self._nav = QWidget()
        nav_layout = QHBoxLayout(self._nav)
        nav_layout.setContentsMargins(12, 8, 12, 8)
        self._back_btn = QPushButton("← Back")
        self._back_btn.clicked.connect(self._go_welcome)
        nav_layout.addWidget(self._back_btn)
        self._nav_title = QLabel()
        self._nav_title.setStyleSheet("font-size: 16px; font-weight: bold;")
        nav_layout.addWidget(self._nav_title)
        nav_layout.addStretch()
        gear_btn = QToolButton()
        gear_btn.setText("⚙")
        gear_btn.setStyleSheet("font-size: 18px; background: transparent; border: none;")
        gear_btn.clicked.connect(self._open_settings)
        nav_layout.addWidget(gear_btn)
        self._nav.hide()
        main_layout.addWidget(self._nav)

        # Stacked pages: 0=welcome, 1=live, 2=profile, 3=preview
        self._stack = QStackedWidget()
        self._welcome_page = _WelcomePage()
        self._welcome_page.quick_start.connect(self._go_live)
        self._welcome_page.high_quality.connect(self._go_profile)
        self._live_page = _LivePage()
        self._profile_page = _ProfilePage()
        self._preview_overlay = _PreviewOverlay()
        self._preview_overlay.stop_requested.connect(self._stop_pipeline)
        self._stack.addWidget(self._welcome_page)   # 0
        self._stack.addWidget(self._live_page)       # 1
        self._stack.addWidget(self._profile_page)    # 2
        self._stack.addWidget(self._preview_overlay) # 3
        main_layout.addWidget(self._stack)

        # Wire up start buttons
        self._live_page.start_btn.clicked.connect(self._start_pipeline)
        self._profile_page.create_btn.clicked.connect(lambda: None)  # wired inside _ProfilePage
        self._profile_page.create_requested.connect(self._create_profile)

        # Internal aliases required by tests
        self._start_btn      = self._live_page.start_btn
        self._cam_combo      = self._live_page.cam_combo
        self._profile_combo  = self._profile_page.profile_combo
        self._quality_ultra  = QCheckBox()  # for test compatibility; real quality in settings
        self._quality_ultra.setChecked(self._settings.get("quality", "ultra") == "ultra")

        # Status bar
        self.setStatusBar(QStatusBar())

        # Frame display timer
        self._frame_timer = QTimer(self)
        self._frame_timer.setInterval(33)
        self._frame_timer.timeout.connect(self._update_preview)

    # --- Navigation ---

    def _go_welcome(self) -> None:
        self._stack.setCurrentIndex(0)
        self._nav.hide()

    def _go_live(self) -> None:
        self._tabs.setCurrentIndex(_ModeIndex.LIVE)
        self._nav_title.setText("Live Mode")
        self._stack.setCurrentIndex(1)
        self._nav.show()

    def _go_profile(self) -> None:
        self._tabs.setCurrentIndex(_ModeIndex.PROFILE)
        self._nav_title.setText("Profile Mode")
        self._stack.setCurrentIndex(2)
        self._nav.show()

    def _open_settings(self) -> None:
        dlg = _SettingsDialog(self._settings, self)
        if dlg.exec():
            self._settings = dlg.result_settings()
            _save_settings(self._settings)
            self._quality_ultra.setChecked(self._settings.get("quality") == "ultra")

    # --- Pipeline start / stop ---

    def _load_models(self) -> None:
        from faceswap.core.swapper  import FaceSwapper
        from faceswap.core.enhancer import GPENEnhancer
        from faceswap.core.parser   import BiSeNetParser
        import onnxruntime as ort

        providers = ort.get_available_providers()
        self._swapper  = FaceSwapper("models/alphaface_256.onnx",  model_type="alphaface")
        self._enhancer = GPENEnhancer("models/GPEN-BFR-512.onnx")
        self._parser   = BiSeNetParser("models/bisenet.onnx")

    def _start_pipeline(self) -> None:
        # NOTE: threading, gc, load_profile_by_name, generate_profile_source_face,
        # SwapPipeline are referenced as module globals so that the AST-based test
        # can inject mocks via its namespace dict without local imports shadowing them.
        if self._starting:
            return
        self._starting = True

        # Profile mode guard  (1 == profile mode index)
        if self._tabs.currentIndex() == 1:
            if (self._profile_combo.count() == 0 or
                    self._profile_combo.currentText() == "(no profiles)"):
                self.statusBar().showMessage("Create a profile first")
                self._starting = False
                return

        self.statusBar().showMessage("Starting…")

        # Clear existing models before sourcing the face (test requirement)
        self._pipeline = None
        self._swapper  = None
        self._enhancer = None
        self._parser   = None

        def _worker():
            try:
                # --- Source face ---
                source_face = None
                if self._tabs.currentIndex() == 1:
                    profile = load_profile_by_name(self._profile_combo.currentText())
                    source_face = generate_profile_source_face(profile)
                else:
                    from faceswap.core.detector import detect_largest as _detect
                    from PIL import Image
                    path = getattr(self, '_live_page', None)
                    photo = path.photo_path if path is not None else ""
                    with Image.open(photo) as img:
                        frame = np.asarray(img.convert("RGB"))[:, :, ::-1].copy()
                    source_face = _detect(frame)
                    if source_face is None:
                        raise ValueError("No face detected in the source photo")

                # --- Models ---
                self._load_models()

                # --- Pipeline ---
                settings = getattr(self, '_settings', {})
                pipeline = SwapPipeline(
                    self._swapper, self._enhancer, self._parser,
                    quality="ultra" if self._quality_ultra.isChecked() else "fast",
                    pixel_boost=settings.get("pixel_boost", False),
                    use_pct=settings.get("pct_harmonize", False),
                    use_occluder=settings.get("occlusion", False),
                )
                pipeline.set_source_face(source_face)

                # --- Camera capture ---
                # sys.modules patching in tests covers faceswap.camera.capture
                from faceswap.camera.capture import CaptureThread
                cam_data = self._cam_combo.currentData()
                cam_idx = cam_data if isinstance(cam_data, int) else 0
                cap = CaptureThread(cam_idx)

                self._signals.startup_ready.emit((pipeline, cap))
            except Exception as exc:
                self._signals.failed.emit("Startup failed", str(exc))

        gc.collect()
        threading.Thread(target=_worker, daemon=True).start()

    @Slot(object)
    def _on_startup_ready(self, payload) -> None:
        pipeline, cap = payload
        self._pipeline = pipeline
        self._cap = cap
        self._running = True
        self._starting = False

        # Virtual camera
        if self._settings.get("vcam_enabled", True):
            try:
                from faceswap.camera.virtual_cam import VirtualCamera
                from pathlib import Path
                dll = str(Path("models/softcam.dll"))
                self._vcam = VirtualCamera(1280, 720, 30.0, dll_path=dll)
            except Exception as e:
                self.statusBar().showMessage(f"VCam unavailable: {e}")
                self._vcam = None

        # WebRTC
        if self._settings.get("webrtc_enabled", False):
            try:
                from faceswap.streaming.webrtc_output import WebRTCStreamer
                streamer = WebRTCStreamer()
                streamer.start()
                pipeline.set_webrtc_streamer(streamer)
                self.statusBar().showMessage(f"WebRTC: {streamer.local_url()}")
            except Exception as e:
                self.statusBar().showMessage(f"WebRTC unavailable: {e}")

        from faceswap.core.pipeline import AsyncPipelineRunner
        self._runner = AsyncPipelineRunner(
            self._pipeline, self._deliver_frame)
        self._runner.start(self._cap)
        self._frame_timer.start()
        self._stack.setCurrentIndex(3)  # show preview

    def _deliver_frame(self, frame: np.ndarray) -> None:
        self._latest_frame = frame
        if self._vcam is not None:
            try:
                resized = cv2.resize(frame, (1280, 720))
                self._vcam.send_frame(resized)
            except Exception:
                pass

    def _update_preview(self) -> None:
        frame = getattr(self, "_latest_frame", None)
        if frame is not None:
            self._preview_overlay.show_frame(frame)
            if self._pipeline:
                ms = self._pipeline.last_inference_ms
                fps = 1000 / ms if ms > 0 else 0
                self._preview_overlay.set_stats(f"{fps:.0f} fps  |  {ms:.0f} ms")

    @Slot(str, str)
    def _on_error(self, title: str, msg: str) -> None:
        self._starting = False
        self.statusBar().showMessage(f"{title}: {msg}")

    @Slot(object, str, float)
    def _on_source_ready(self, face, filename: str, confidence: float) -> None:
        self.statusBar().showMessage(f"Source: {filename}  ({confidence:.0f}%)")

    @Slot(str, float)
    def _on_profile_done(self, name: str, quality: float) -> None:
        self._profile_page.show_done(name, quality)

    @Slot(str)
    def _on_progress_msg(self, msg: str) -> None:
        self._profile_page.show_progress(msg)

    def _stop_pipeline(self) -> bool:
        """Stop the running pipeline. Returns True on clean stop, False on timeout."""
        if not self._running:
            return True
        try:
            if self._runner is not None:
                self._runner.stop()
                self._runner.join()
        except TimeoutError:
            self.statusBar().showMessage("Stop timed out — retrying…")
            return False

        if self._vcam is not None:
            self._vcam.close()
        self._vcam = None
        self._runner = None
        self._pipeline = None
        self._running = False

        ft = getattr(self, '_frame_timer', None)
        if ft is not None:
            ft.stop()
        # Return to the page we came from
        stk = getattr(self, '_stack', None)
        tabs = getattr(self, '_tabs', None)
        if stk is not None and tabs is not None:
            stk.setCurrentIndex(2 if tabs.currentIndex() == 1 else 1)
        self.statusBar().showMessage("Stopped")
        return True

    # --- Profile creation ---

    def _create_profile(self, paths: list[str], name: str) -> None:
        def _worker():
            try:
                from faceswap.profile.manager import create_profile_level1
                profile = create_profile_level1(
                    name, paths,
                    progress_callback=lambda m: self._signals.progress_msg.emit(m),
                )
                self._signals.profile_done.emit(name, profile.quality_score)
            except Exception as exc:
                self._signals.failed.emit("Profile creation failed", str(exc))

        import threading as _threading
        _threading.Thread(target=_worker, daemon=True).start()

    # --- Lifecycle ---

    def closeEvent(self, event):
        self._closing = True
        self._stop_pipeline()
        _save_settings(self._settings)
        event.accept()
