"""
Main application window (PySide6).

Layout: a fixed sidebar (Live, Profiles, Settings) beside a stack of pages.
Every page has a header row with its title and primary action, then content.
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
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal, QObject, Slot, QSize
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFileDialog, QProgressBar, QCheckBox, QLineEdit,
    QStatusBar, QSizePolicy, QStackedWidget, QScrollArea, QFrame,
    QButtonGroup,
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
    "tensorrt": False,
    "last_photo": "",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

_CONSENT_TEXT = "I have this person's permission to use their face"

# Stack page indices
_PAGE_LIVE, _PAGE_PROFILES, _PAGE_PREVIEW, _PAGE_SETTINGS = range(4)

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
# Theme: near-black surfaces, hairline borders, one indigo accent, 13px type
# ---------------------------------------------------------------------------

_ACCENT = "#5e6ad2"

_STYLE = """
* {
    font-family: "Inter", "Segoe UI Variable Text", "Segoe UI", sans-serif;
    font-size: 13px;
    color: #e6e7ea;
}
QMainWindow { background: #101113; }
QFrame#sidebar { background: #08090a; border: none; border-right: 1px solid #1f2023; }
QFrame#header { background: #101113; border: none; border-bottom: 1px solid #1f2023; }
QFrame#card { background: #131416; border: 1px solid #23252a; border-radius: 8px; }
QFrame#divider { background: #1f2023; border: none; max-height: 1px; min-height: 1px; }
QStackedWidget, QScrollArea, QWidget#page { background: #101113; border: none; }

QLabel { background: transparent; }
QLabel#brand { font-size: 14px; font-weight: 600; color: #f7f8f8; }
QLabel#page_title { font-size: 15px; font-weight: 600; color: #f7f8f8; }
QLabel#muted, QLabel#page_subtitle { color: #8a8f98; font-size: 12px; }
QLabel#section { color: #8a8f98; font-size: 12px; font-weight: 600; }
QLabel#row_title { color: #f7f8f8; }
QLabel#pill {
    background: #1b1c1f; border: 1px solid #2a2c31; border-radius: 9px;
    padding: 1px 8px; color: #b4b8bf; font-size: 12px;
}
QLabel#video { background: #000000; border: 1px solid #23252a; border-radius: 8px; }
QLabel#thumb { background: #1b1c1f; border: 1px solid #2a2c31; border-radius: 6px; }

QPushButton {
    background: #1b1c1f; border: 1px solid #2a2c31; border-radius: 6px;
    padding: 5px 12px; min-height: 18px;
}
QPushButton:hover { background: #232428; border-color: #34363c; }
QPushButton:pressed { background: #17181a; }
QPushButton:disabled { color: #62666d; background: #141517; border-color: #1f2023; }
QPushButton#primary {
    background: #5e6ad2; border: 1px solid #6c78e6; color: #ffffff; font-weight: 600;
}
QPushButton#primary:hover { background: #6c78e6; }
QPushButton#primary:disabled { background: #25273d; border-color: #25273d; color: #7b7f9c; }
QPushButton#danger { background: transparent; border: 1px solid #5c2b2e; color: #eb5757; }
QPushButton#danger:hover { background: #2a1517; }
QPushButton#nav {
    background: transparent; border: none; border-radius: 6px;
    padding: 6px 10px; text-align: left; color: #8a8f98;
}
QPushButton#nav:hover { background: #151618; color: #e6e7ea; }
QPushButton#nav:checked { background: #1b1c1f; color: #f7f8f8; }
QPushButton#row {
    background: transparent; border: none; border-radius: 6px;
    padding: 8px 10px; text-align: left;
}
QPushButton#row:hover { background: #1b1c1f; }
QPushButton#row:checked { background: #1e2030; color: #f7f8f8; }
QPushButton#segment { border-radius: 0; padding: 4px 14px; color: #8a8f98; }
QPushButton#segment:checked { background: #2a2c31; color: #f7f8f8; }

QComboBox, QLineEdit {
    background: #1b1c1f; border: 1px solid #2a2c31; border-radius: 6px;
    padding: 5px 10px; min-height: 18px; selection-background-color: #5e6ad2;
}
QComboBox:hover, QLineEdit:hover { border-color: #34363c; }
QComboBox:focus, QLineEdit:focus { border-color: #5e6ad2; }
QComboBox QAbstractItemView {
    background: #1b1c1f; border: 1px solid #2a2c31; outline: none;
    selection-background-color: #2a2c31;
}

QCheckBox { spacing: 8px; background: transparent; }
QCheckBox::indicator {
    width: 14px; height: 14px; border-radius: 4px;
    border: 1px solid #3a3d44; background: #1b1c1f;
}
QCheckBox::indicator:hover { border-color: #5e6ad2; }
QCheckBox::indicator:checked { background: #5e6ad2; border-color: #6c78e6; }

QProgressBar {
    background: #1b1c1f; border: none; border-radius: 2px;
    max-height: 4px; min-height: 4px;
}
QProgressBar::chunk { background: #5e6ad2; border-radius: 2px; }

QFrame#drop_zone { border: 1px dashed #34363c; border-radius: 8px; background: #121315; }
QFrame#drop_zone:hover { border-color: #5e6ad2; background: #14151a; }

QStatusBar {
    background: #08090a; border-top: 1px solid #1f2023; color: #8a8f98; font-size: 12px;
}
QStatusBar QLabel { color: #8a8f98; font-size: 12px; }
QScrollBar:vertical { background: transparent; width: 8px; margin: 0; }
QScrollBar::handle:vertical { background: #2a2c31; border-radius: 4px; min-height: 24px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""


def _label(text: str, name: str = "", wrap: bool = False) -> QLabel:
    label = QLabel(text)
    if name:
        label.setObjectName(name)
    label.setWordWrap(wrap)
    return label


def _read_image(path: str) -> np.ndarray | None:
    """cv2.imread that also works for non-ASCII paths on Windows."""
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        return None


def _pixmap_from_bgr(bgr: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    h, w = rgb.shape[:2]
    # QPixmap.fromImage copies the pixels, so the numpy buffer may go away.
    return QPixmap.fromImage(QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888))


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
# Small building blocks
# ---------------------------------------------------------------------------

class _Toggle(QCheckBox):
    """A switch: QCheckBox behaviour, painted as a pill with a sliding knob."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(30, 18)

    def sizeHint(self) -> QSize:
        return QSize(30, 18)

    def hitButton(self, pos) -> bool:
        return self.rect().contains(pos)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(_ACCENT if self.isChecked() else "#2a2c31"))
        painter.drawRoundedRect(0, 0, 30, 18, 9, 9)
        painter.setBrush(QColor("#ffffff" if self.isChecked() else "#8a8f98"))
        painter.drawEllipse(14 if self.isChecked() else 2, 2, 14, 14)
        painter.end()


class _PageHeader(QFrame):
    """Title and subtitle on the left, the page's actions on the right."""

    def __init__(self, title: str, subtitle: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("header")
        self.setFixedHeight(56)
        row = QHBoxLayout(self)
        row.setContentsMargins(24, 0, 24, 0)
        row.setSpacing(8)
        titles = QVBoxLayout()
        titles.setSpacing(1)
        titles.setAlignment(Qt.AlignVCenter)
        titles.addWidget(_label(title, "page_title"))
        if subtitle:
            titles.addWidget(_label(subtitle, "page_subtitle"))
        row.addLayout(titles)
        row.addStretch()
        self.actions = QHBoxLayout()
        self.actions.setSpacing(8)
        row.addLayout(self.actions)


def _page(header: _PageHeader) -> tuple[QWidget, QVBoxLayout]:
    """A page: header, then a scrollable content column. Returns (page, column)."""
    page = QWidget()
    page.setObjectName("page")
    outer = QVBoxLayout(page)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)
    outer.addWidget(header)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    body = QWidget()
    body.setObjectName("page")
    column = QVBoxLayout(body)
    column.setContentsMargins(24, 20, 24, 24)
    column.setSpacing(10)
    column.setAlignment(Qt.AlignTop)
    body.setMaximumWidth(720)
    scroll.setWidget(body)
    outer.addWidget(scroll)
    return page, column


def _divider() -> QFrame:
    line = QFrame()
    line.setObjectName("divider")
    return line


class _DropZone(QFrame):
    files_dropped = Signal(list)

    def __init__(self, text: str, multiple: bool = False, parent=None):
        super().__init__(parent)
        self.setObjectName("drop_zone")
        self._multiple = multiple
        self.setAcceptDrops(True)
        self.setMinimumHeight(96)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        self._label = _label(text, "muted", wrap=True)
        self._label.setAlignment(Qt.AlignCenter)
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


class _PhotoStrip(QWidget):
    def __init__(self, max_photos: int = 10, parent=None):
        super().__init__(parent)
        self._max = max_photos
        self._paths: list[str] = []
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self._cells: list[QLabel] = []
        for _ in range(max_photos):
            cell = _label("", "thumb")
            cell.setFixedSize(48, 48)
            cell.setAlignment(Qt.AlignCenter)
            layout.addWidget(cell)
            self._cells.append(cell)
        layout.addStretch()

    def set_paths(self, paths: list[str]) -> None:
        self._paths = paths[:self._max]
        for i, cell in enumerate(self._cells):
            cell.setPixmap(QPixmap())
            cell.setText("")
            if i < len(self._paths):
                img = _read_image(self._paths[i])
                if img is not None:
                    cell.setPixmap(_pixmap_from_bgr(cv2.resize(img, (46, 46))))
                else:
                    cell.setText("?")


# ---------------------------------------------------------------------------
# Live page: one photo, pick a camera, start
# ---------------------------------------------------------------------------

class _LivePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        header = _PageHeader("Live", "One source photo, start immediately")
        self.start_btn = QPushButton("Start")
        self.start_btn.setObjectName("primary")
        self.start_btn.setEnabled(False)
        header.actions.addWidget(self.start_btn)

        page, column = _page(header)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(page)

        column.addWidget(_label("Source face", "section"))
        source_row = QHBoxLayout()
        source_row.setSpacing(12)
        self._photo_label = _label("", "thumb")
        self._photo_label.setFixedSize(96, 96)
        self._photo_label.setAlignment(Qt.AlignCenter)
        self._photo_label.hide()
        source_row.addWidget(self._photo_label)
        self._drop = _DropZone("Drop a photo here, or click to browse", multiple=False)
        self._drop.files_dropped.connect(self._on_photo)
        source_row.addWidget(self._drop, stretch=1)
        column.addLayout(source_row)

        self.consent_cb = QCheckBox(_CONSENT_TEXT)
        self.consent_cb.toggled.connect(self._update_start)
        column.addWidget(self.consent_cb)

        column.addSpacing(10)
        column.addWidget(_label("Camera", "section"))
        self.cam_combo = QComboBox()
        self.cam_combo.setMaximumWidth(280)
        self._populate_cameras()
        column.addWidget(self.cam_combo)

        column.addSpacing(10)
        column.addWidget(_divider())
        column.addWidget(_label(
            "After starting, choose the virtual camera named in the status bar in your video app. "
            "Every swapped frame carries a visible \"AI-GENERATED FACE\" label.",
            "muted", wrap=True))

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

    def _update_start(self) -> None:
        self.start_btn.setEnabled(bool(self._photo_path) and self.consent_cb.isChecked())

    def _on_photo(self, paths: list[str]) -> None:
        self._photo_path = paths[0]
        # Permission is given per person: a new photo needs it again.
        self.consent_cb.setChecked(False)
        img = _read_image(self._photo_path)
        if img is not None:
            h, w = img.shape[:2]
            scale = 94 / max(h, w, 1)
            small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
            self._photo_label.setPixmap(_pixmap_from_bgr(small))
            self._photo_label.show()
        self._drop.set_text(f"{Path(self._photo_path).name}\nDrop or click to replace")
        self._update_start()

    @property
    def photo_path(self) -> str:
        return self._photo_path

    @property
    def camera_index(self) -> int:
        d = self.cam_combo.currentData()
        return d if d is not None else 0


# ---------------------------------------------------------------------------
# Profiles page: pick a saved profile and start, or create a new one
# ---------------------------------------------------------------------------

class _ProfilePage(QWidget):
    create_requested = Signal(list, str)   # paths, name

    def __init__(self, parent=None):
        super().__init__(parent)
        header = _PageHeader("Profiles", "Several photos of one person, stronger identity")
        self.start_btn = QPushButton("Start")
        self.start_btn.setObjectName("primary")
        self.start_btn.setEnabled(False)
        header.actions.addWidget(self.start_btn)

        page, column = _page(header)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(page)

        # Hidden selector: _start_pipeline reads the chosen profile from it.
        self.profile_combo = QComboBox()
        self.profile_combo.hide()
        column.addWidget(self.profile_combo)

        column.addWidget(_label("Saved profiles", "section"))
        card = QFrame()
        card.setObjectName("card")
        self._profiles_layout = QVBoxLayout(card)
        self._profiles_layout.setContentsMargins(6, 6, 6, 6)
        self._profiles_layout.setSpacing(2)
        self._profile_group = QButtonGroup(self)
        self._profile_group.setExclusive(True)
        column.addWidget(card)

        self.consent_cb = QCheckBox(_CONSENT_TEXT)
        self.consent_cb.toggled.connect(self._update_buttons)
        column.addWidget(self.consent_cb)

        column.addSpacing(10)
        column.addWidget(_divider())
        column.addSpacing(4)
        column.addWidget(_label("New profile", "section"))

        self._drop = _DropZone("Drop 5-10 photos of the person, or click to browse",
                               multiple=True)
        self._drop.files_dropped.connect(self._on_photos)
        column.addWidget(self._drop)

        self._strip = _PhotoStrip(max_photos=10)
        column.addWidget(self._strip)

        name_row = QHBoxLayout()
        name_row.setSpacing(8)
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Profile name")
        self._name_edit.setMaximumWidth(280)
        self._name_edit.textChanged.connect(self._update_buttons)
        name_row.addWidget(self._name_edit)
        self.create_btn = QPushButton("Create profile")
        self.create_btn.setEnabled(False)
        self.create_btn.clicked.connect(self._on_create)
        name_row.addWidget(self.create_btn)
        name_row.addStretch()
        column.addLayout(name_row)

        self._progress = QProgressBar()
        self._progress.setTextVisible(False)
        self._progress.hide()
        self._progress_label = _label("", "muted", wrap=True)
        self._progress_label.hide()
        column.addWidget(self._progress)
        column.addWidget(self._progress_label)

        self._photo_paths: list[str] = []
        self._creating = False
        self.refresh_profiles()

    def _has_profile(self) -> bool:
        return (self.profile_combo.count() > 0
                and self.profile_combo.currentText() != "(no profiles)")

    def _update_buttons(self) -> None:
        consent = self.consent_cb.isChecked()
        self.start_btn.setEnabled(consent and self._has_profile())
        self.create_btn.setEnabled(
            consent and not self._creating and bool(self._photo_paths)
            and bool(self._name_edit.text().strip()))

    def _on_photos(self, paths: list[str]) -> None:
        self._photo_paths = paths[:10]
        self._strip.set_paths(self._photo_paths)
        self._drop.set_text(f"{len(self._photo_paths)} of 10 photos selected\n"
                            "Drop or click to replace")
        self._update_buttons()

    def _on_create(self) -> None:
        name = self._name_edit.text().strip()
        if not name or not self._photo_paths:
            return
        self._creating = True
        self._update_buttons()
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.show()
        self._progress_label.setText("Analyzing photos…")
        self._progress_label.show()
        self.create_requested.emit(self._photo_paths, name)

    def show_progress(self, msg: str) -> None:
        self._progress_label.setText(msg)

    def show_failed(self) -> None:
        """Creation failed: stop the spinner and let the user retry."""
        self._creating = False
        self._progress.hide()
        self._progress_label.hide()
        self._update_buttons()

    def show_done(self, name: str, quality: float) -> None:
        self._creating = False
        self._progress.hide()
        self._progress_label.setText(
            f"Profile \"{name}\" created. Identity strength: {int(quality * 100)}%")
        self._name_edit.clear()
        self._photo_paths = []
        self._strip.set_paths([])
        self._drop.set_text("Drop 5-10 photos of the person, or click to browse")
        self.refresh_profiles()
        self._select_profile(name)

    def refresh_profiles(self) -> None:
        try:
            from faceswap.profile.manager import list_profiles
            profiles = list(list_profiles())
        except Exception:
            profiles = []

        self.profile_combo.clear()
        if not profiles:
            self.profile_combo.addItem("(no profiles)")
        else:
            for name in profiles:
                self.profile_combo.addItem(name)

        while self._profiles_layout.count():
            item = self._profiles_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                self._profile_group.removeButton(widget)
                widget.deleteLater()

        if not profiles:
            empty = _label("No profiles yet. Create one below.", "muted")
            empty.setContentsMargins(10, 8, 10, 8)
            self._profiles_layout.addWidget(empty)
        else:
            for index, name in enumerate(profiles):
                row = QPushButton(name)
                row.setObjectName("row")
                row.setCheckable(True)
                row.setChecked(index == 0)
                row.setCursor(Qt.PointingHandCursor)
                row.clicked.connect(lambda checked=False, n=name: self._select_profile(n))
                self._profile_group.addButton(row)
                self._profiles_layout.addWidget(row)
        self._update_buttons()

    def _select_profile(self, name: str) -> None:
        # Sync hidden combo so _start_pipeline picks up the right profile
        idx = self.profile_combo.findText(name)
        if idx >= 0:
            self.profile_combo.setCurrentIndex(idx)
            for button in self._profile_group.buttons():
                if button.text() == name:
                    button.setChecked(True)
        self._update_buttons()


# ---------------------------------------------------------------------------
# Settings page: rows of label + control, saved as soon as they change
# ---------------------------------------------------------------------------

class _SettingsPage(QWidget):
    changed = Signal(dict)

    _TOGGLES = [
        # (settings key, title, description)
        ("vcam_enabled", "Virtual camera",
         "Send output to the DirectShow Softcam device."),
        ("webrtc_enabled", "Stream to phone",
         "Serve the output over WebRTC on port 8765."),
        ("pixel_boost", "Pixel boost",
         "Swap at 512 px in four passes. Sharper, about four times the swap cost."),
        ("pct_harmonize", "Local color harmonization",
         "Match lighting that differs across the face."),
        ("occlusion", "Occlusion detection",
         "Keep hands and objects in front of the face."),
        ("tensorrt", "TensorRT for restoration",
         "Experimental and untested. Needs TensorRT installed; the first start "
         "builds an engine and takes several minutes."),
    ]

    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)
        self._s = dict(settings)
        header = _PageHeader("Settings", "Changes apply the next time you start")
        page, column = _page(header)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(page)

        card = QFrame()
        card.setObjectName("card")
        rows = QVBoxLayout(card)
        rows.setContentsMargins(16, 6, 16, 6)
        rows.setSpacing(0)

        # Quality: a two-way segmented control
        segment = QHBoxLayout()
        segment.setSpacing(0)
        self._quality_fast = QPushButton("Fast")
        self._quality_ultra = QPushButton("Ultra")
        group = QButtonGroup(self)
        group.setExclusive(True)
        for button in (self._quality_fast, self._quality_ultra):
            button.setObjectName("segment")
            button.setCheckable(True)
            group.addButton(button)
            segment.addWidget(button)
        ultra = self._s.get("quality", "ultra") == "ultra"
        self._quality_ultra.setChecked(ultra)
        self._quality_fast.setChecked(not ultra)
        group.buttonClicked.connect(lambda _button: self._emit())
        control = QWidget()
        control.setLayout(segment)
        rows.addLayout(self._row(
            "Quality", "Ultra adds skin detail, scattering and highlight transfer.", control))

        self._toggles: dict[str, _Toggle] = {}
        for key, title, description in self._TOGGLES:
            rows.addWidget(_divider())
            toggle = _Toggle()
            toggle.setChecked(bool(self._s.get(key, _DEFAULT_SETTINGS[key])))
            toggle.toggled.connect(lambda _checked: self._emit())
            self._toggles[key] = toggle
            rows.addLayout(self._row(title, description, toggle))
        column.addWidget(card)

    @staticmethod
    def _row(title: str, description: str, control: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 10, 0, 10)
        row.setSpacing(16)
        text = QVBoxLayout()
        text.setSpacing(2)
        text.addWidget(_label(title, "row_title"))
        text.addWidget(_label(description, "muted", wrap=True))
        row.addLayout(text, stretch=1)
        row.addWidget(control, alignment=Qt.AlignVCenter)
        return row

    def result_settings(self) -> dict:
        result = dict(self._s, quality="ultra" if self._quality_ultra.isChecked() else "fast")
        for key, toggle in self._toggles.items():
            result[key] = toggle.isChecked()
        return result

    def _emit(self) -> None:
        self._s = self.result_settings()
        self.changed.emit(dict(self._s))


# ---------------------------------------------------------------------------
# Preview page (shown while the pipeline is running)
# ---------------------------------------------------------------------------

class _PreviewOverlay(QWidget):
    stop_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = _PageHeader("Live output")
        self._stats = _label("Starting", "pill")
        header.actions.addWidget(self._stats, alignment=Qt.AlignVCenter)
        stop_btn = QPushButton("Stop")
        stop_btn.setObjectName("danger")
        stop_btn.clicked.connect(self.stop_requested)
        header.actions.addWidget(stop_btn)
        outer.addWidget(header)

        body = QVBoxLayout()
        body.setContentsMargins(24, 20, 24, 16)
        body.setSpacing(10)
        self._video = _label("", "video")
        self._video.setAlignment(Qt.AlignCenter)
        self._video.setMinimumSize(320, 180)
        self._video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        body.addWidget(self._video, stretch=1)
        self._stages = _label("", "muted", wrap=True)
        body.addWidget(self._stages)
        outer.addLayout(body, stretch=1)

    def show_frame(self, bgr: np.ndarray) -> None:
        self._video.setPixmap(_pixmap_from_bgr(bgr).scaled(
            self._video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def set_stats(self, text: str) -> None:
        self._stats.setText(text)

    def set_stages(self, text: str) -> None:
        self._stages.setText(text)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("liveswap")
        self.setMinimumSize(960, 640)
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
        self._shown_errors = 0

        # --- Mode tracker (satisfies _tabs.currentIndex() contract) ---
        self._tabs = _ModeIndex()

        # --- Build UI: sidebar | pages ---
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(208)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(10, 14, 10, 10)
        side.setSpacing(2)
        brand = _label("liveswap", "brand")
        brand.setContentsMargins(10, 0, 0, 10)
        side.addWidget(brand)
        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)
        self._nav_live = self._nav_button("Live", self._go_live)
        self._nav_profiles = self._nav_button("Profiles", self._go_profile)
        side.addWidget(self._nav_live)
        side.addWidget(self._nav_profiles)
        side.addStretch()
        self._nav_settings = self._nav_button("Settings", self._open_settings)
        side.addWidget(self._nav_settings)
        main_layout.addWidget(sidebar)

        self._stack = QStackedWidget()
        self._live_page = _LivePage()
        self._profile_page = _ProfilePage()
        self._preview_overlay = _PreviewOverlay()
        self._preview_overlay.stop_requested.connect(self._stop_pipeline)
        self._settings_page = _SettingsPage(self._settings)
        self._settings_page.changed.connect(self._on_settings_changed)
        self._stack.addWidget(self._live_page)         # _PAGE_LIVE
        self._stack.addWidget(self._profile_page)      # _PAGE_PROFILES
        self._stack.addWidget(self._preview_overlay)   # _PAGE_PREVIEW
        self._stack.addWidget(self._settings_page)     # _PAGE_SETTINGS
        main_layout.addWidget(self._stack, stretch=1)

        # Wire up start buttons
        self._live_page.start_btn.clicked.connect(self._on_start_clicked)
        self._profile_page.start_btn.clicked.connect(self._on_start_clicked)
        self._profile_page.create_requested.connect(self._create_profile)

        # Internal aliases required by tests
        self._start_btn      = self._live_page.start_btn
        self._cam_combo      = self._live_page.cam_combo
        self._profile_combo  = self._profile_page.profile_combo
        self._quality_ultra  = QCheckBox()  # for test compatibility; real quality in settings
        self._quality_ultra.setChecked(self._settings.get("quality", "ultra") == "ultra")

        # Status bar
        self.setStatusBar(QStatusBar())
        self.statusBar().setSizeGripEnabled(False)

        # Frame display timer
        self._frame_timer = QTimer(self)
        self._frame_timer.setInterval(33)
        self._frame_timer.timeout.connect(self._update_preview)

        self._nav_live.setChecked(True)
        self._go_live()

    def _nav_button(self, text: str, handler) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName("nav")
        button.setCheckable(True)
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(lambda _checked=False: handler())
        self._nav_group.addButton(button)
        return button

    # --- Navigation ---

    def _go_live(self) -> None:
        if not self._running:
            self._tabs.setCurrentIndex(_ModeIndex.LIVE)
        # While running, both mode pages are replaced by the live output.
        self._stack.setCurrentIndex(_PAGE_PREVIEW if self._running else _PAGE_LIVE)

    def _go_profile(self) -> None:
        if not self._running:
            self._tabs.setCurrentIndex(_ModeIndex.PROFILE)
        self._stack.setCurrentIndex(_PAGE_PREVIEW if self._running else _PAGE_PROFILES)

    def _open_settings(self) -> None:
        self._stack.setCurrentIndex(_PAGE_SETTINGS)

    def _on_settings_changed(self, settings: dict) -> None:
        self._settings = settings
        _save_settings(self._settings)
        self._quality_ultra.setChecked(self._settings.get("quality") == "ultra")

    # --- Pipeline start / stop ---

    def _on_start_clicked(self) -> None:
        """Start only with the user's confirmation that the source face may be used."""
        profile_mode = self._tabs.currentIndex() == _ModeIndex.PROFILE
        page = self._profile_page if profile_mode else self._live_page
        if not page.consent_cb.isChecked():
            self.statusBar().showMessage("Confirm you have permission to use this face")
            return
        self._start_pipeline()

    def _load_models(self) -> None:
        from faceswap.core.swapper  import load_default_swapper
        from faceswap.core.enhancer import FaceEnhancer
        from faceswap.core.parser   import FaceParser

        self._swapper  = load_default_swapper("models")
        self._enhancer = FaceEnhancer("models/GPEN-BFR-512.onnx",
                                      tensorrt=self._settings.get("tensorrt", False))
        self._parser   = FaceParser("models/bisenet.onnx")

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
        self._shown_errors = 0
        self.statusBar().showMessage("Running")

        # Virtual camera
        if self._settings.get("vcam_enabled", True):
            try:
                from faceswap.camera.virtual_cam import VirtualCamera
                dll = str(Path("models/softcam.dll"))
                self._vcam = VirtualCamera(1280, 720, 30.0, dll_path=dll)
                self.statusBar().showMessage(
                    f"Running. Camera name in video apps: {self._vcam.device_name}")
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
        self._stack.setCurrentIndex(_PAGE_PREVIEW)

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
                self._preview_overlay.set_stats(f"{fps:.0f} fps  ·  {ms:.0f} ms")
                slowest = sorted(self._pipeline.stage_ms.items(),
                                 key=lambda item: item[1], reverse=True)[:6]
                self._preview_overlay.set_stages(
                    "   ".join(f"{name} {stage_ms:.0f} ms" for name, stage_ms in slowest))
        # Frames that fail in the pipeline are passed through unprocessed; say so.
        runner = self._runner
        if runner is not None and runner.error_count != self._shown_errors:
            self._shown_errors = runner.error_count
            self.statusBar().showMessage(
                f"{runner.error_count} frame(s) failed and were shown unprocessed: "
                f"{runner.last_error}")

    @Slot(str, str)
    def _on_error(self, title: str, msg: str) -> None:
        self._starting = False
        self._profile_page.show_failed()
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
            stk.setCurrentIndex(1 if tabs.currentIndex() == 1 else 0)
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

        threading.Thread(target=_worker, daemon=True).start()

    # --- Lifecycle ---

    def closeEvent(self, event):
        self._closing = True
        self._stop_pipeline()
        _save_settings(self._settings)
        event.accept()
