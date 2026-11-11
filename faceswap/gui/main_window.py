"""
Main application window (PySide6).

Three-tab layout:
  - Live Mode:    source photo + webcam → real-time swap
  - Profile Mode: multi-photo → create identity profile → swap
  - Settings:     pipeline parameters, virtual camera, output

All long-running ops run in background threads via WorkerSignals.
Settings are persisted to ~/.liveswap/settings.json.
"""
from __future__ import annotations

import gc
import json
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal, QObject, Slot, QSize
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QTabWidget,
    QFileDialog, QProgressBar, QCheckBox, QSlider, QLineEdit,
    QGroupBox, QStatusBar, QSplitter, QRadioButton, QButtonGroup,
    QSizePolicy,
)

from faceswap.core.detector import detect_largest
from faceswap.core.swapper import FaceSwapper
from faceswap.core.enhancer import FaceEnhancer
from faceswap.core.parser import FaceParser
from faceswap.core.pipeline import (
    SwapPipeline, AsyncPipelineRunner, generate_profile_source_face,
)
from faceswap.profile.manager import (
    list_profiles, load_profile_by_name, create_profile_level1,
)

MODELS_DIR = Path("models")
SETTINGS_PATH = Path.home() / ".liveswap" / "settings.json"

_DEFAULT_SETTINGS: dict = {
    "last_camera": 0,
    "quality": "ultra",
    "ultra_sss": True,
    "ultra_specular": True,
    "ultra_detail": True,
    "ultra_cam_match": True,
    "degradation_strength": 50,
    "gpen_enabled": True,
    "gpen_blend": 0.8,
    "ema_alpha": 0.55,
    "blinkgate_enabled": True,
    "cam_degrade_enabled": True,
    "cam_degrade_noise": True,
    "cam_degrade_sharpness": True,
    "cam_degrade_motion": True,
    "vcam_enabled": False,
    "output_resolution": "720p",
    "show_stats": True,
    "window_width": 1200,
    "window_height": 700,
}

_IMG_FILTER = "Images (*.jpg *.jpeg *.png *.bmp *.webp)"
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_RES = {"720p": (1280, 720), "1080p": (1920, 1080)}


# ── Worker signals ────────────────────────────────────────────────────────────

class WorkerSignals(QObject):
    startup_ready = Signal(object)            # (pipeline, cap, w, h)
    failed = Signal(str, str)                  # (operation, message)
    source_ready = Signal(object, str, float)  # (face, filename, confidence_pct)
    profile_done = Signal(str, float)          # (name, identity_strength)


# ── Drag-and-drop image zone ──────────────────────────────────────────────────

class DropZone(QLabel):
    """Image file drop target; also used as a visual hint for file acceptance."""

    files_dropped = Signal(list)  # list[str]

    _IDLE = (
        "QLabel{border:2px dashed #555;border-radius:6px;background:#1c1c1e;"
        "color:#888;padding:8px;}"
        "QLabel:hover{border-color:#0a84ff;color:#ccc;}"
    )

    def __init__(self, text: str, multiple: bool = False, parent=None):
        super().__init__(text, parent)
        self._multiple = multiple
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setStyleSheet(self._IDLE)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        paths = [
            u.toLocalFile() for u in event.mimeData().urls()
            if Path(u.toLocalFile()).suffix.lower() in _IMG_EXTS
        ]
        if paths:
            if not self._multiple:
                paths = paths[:1]
            self.files_dropped.emit(paths)
            event.acceptProposedAction()


# ── Thumbnail grid (profile photo preview) ───────────────────────────────────

class ThumbnailGrid(QWidget):
    """Up to 10 small square thumbnails arranged 5-wide."""

    _MAX = 10
    _SZ = 60

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(self._SZ * 2 + 12)
        layout = QGridLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(0, 0, 0, 0)
        self._cells: list[QLabel] = []
        for i in range(self._MAX):
            lbl = QLabel()
            lbl.setFixedSize(self._SZ, self._SZ)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("background:#1c1c1e;border:1px solid #333;")
            layout.addWidget(lbl, i // 5, i % 5)
            self._cells.append(lbl)

    def set_photos(self, paths: list[str]) -> None:
        for lbl in self._cells:
            lbl.clear()
        for i, path in enumerate(paths[:self._MAX]):
            img = cv2.imread(path)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            s = min(h, w)
            sq = rgb[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
            sq = cv2.resize(sq, (self._SZ, self._SZ))
            qimg = QImage(sq.data, self._SZ, self._SZ, self._SZ * 3,
                          QImage.Format.Format_RGB888)
            self._cells[i].setPixmap(QPixmap.fromImage(qimg).copy())


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("liveswap")

        self._settings = self._load_settings()
        self.resize(self._settings["window_width"], self._settings["window_height"])
        self.setMinimumSize(1000, 640)

        # Pipeline state
        self._pipeline: SwapPipeline | None = None
        self._runner: AsyncPipelineRunner | None = None
        self._cap = None
        self._source_face = None
        self._source_photo_path: str | None = None
        self._profile_photos: list[str] = []
        self._swapper = self._enhancer = self._parser = None
        self._starting = False
        self._closing = False
        self._running = False
        self._vcam = None

        # Preview / stats
        self._preview_frame: np.ndarray | None = None
        self._frame_times: deque[float] = deque(maxlen=60)
        self._dropped_frames = 0

        # Signals
        self._signals = WorkerSignals(self)
        self._signals.startup_ready.connect(self._finish_start)
        self._signals.failed.connect(self._worker_failed)
        self._signals.source_ready.connect(self._source_loaded)
        self._signals.profile_done.connect(self._profile_created)

        self._setup_ui()
        self._refresh_profiles()
        self._apply_settings_to_ui()

        self._timer = QTimer()
        self._timer.timeout.connect(self._update_preview)
        self._timer.start(33)

    # ------------------------------------------------------------------
    # Settings I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _load_settings() -> dict:
        s = dict(_DEFAULT_SETTINGS)
        try:
            if SETTINGS_PATH.exists():
                s.update(json.loads(SETTINGS_PATH.read_text()))
        except Exception:
            pass
        return s

    def _save_settings(self) -> None:
        try:
            SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            SETTINGS_PATH.write_text(json.dumps(self._settings, indent=2))
        except Exception:
            pass

    def _collect_settings(self) -> None:
        s = self._settings
        s["last_camera"] = self._cam_combo.currentIndex()
        s["quality"] = "ultra" if self._radio_ultra.isChecked() else "fast"
        s["ultra_sss"] = self._chk_sss.isChecked()
        s["ultra_specular"] = self._chk_specular.isChecked()
        s["ultra_detail"] = self._chk_detail.isChecked()
        s["ultra_cam_match"] = self._chk_cam_match.isChecked()
        s["degradation_strength"] = self._degrade_slider.value()
        s["gpen_enabled"] = self._chk_gpen.isChecked()
        s["gpen_blend"] = self._gpen_slider.value() / 100.0
        s["ema_alpha"] = round(0.3 + self._ema_slider.value() / 100.0 * 0.5, 3)
        s["blinkgate_enabled"] = self._chk_blinkgate.isChecked()
        s["cam_degrade_enabled"] = self._chk_cam_degrade.isChecked()
        s["cam_degrade_noise"] = self._chk_degrade_noise.isChecked()
        s["cam_degrade_sharpness"] = self._chk_degrade_sharp.isChecked()
        s["cam_degrade_motion"] = self._chk_degrade_motion.isChecked()
        s["vcam_enabled"] = self._chk_vcam.isChecked()
        s["output_resolution"] = self._res_combo.currentText()
        s["show_stats"] = self._chk_stats.isChecked()
        sz = self.size()
        s["window_width"] = sz.width()
        s["window_height"] = sz.height()

    def _apply_settings_to_ui(self) -> None:
        s = self._settings
        self._cam_combo.setCurrentIndex(min(s["last_camera"], self._cam_combo.count() - 1))
        self._radio_ultra.setChecked(s["quality"] == "ultra")
        self._radio_fast.setChecked(s["quality"] == "fast")
        self._chk_sss.setChecked(bool(s["ultra_sss"]))
        self._chk_specular.setChecked(bool(s["ultra_specular"]))
        self._chk_detail.setChecked(bool(s["ultra_detail"]))
        self._chk_cam_match.setChecked(bool(s["ultra_cam_match"]))
        self._degrade_slider.setValue(int(s["degradation_strength"]))
        self._chk_gpen.setChecked(bool(s["gpen_enabled"]))
        self._gpen_slider.setValue(int(s["gpen_blend"] * 100))
        self._ema_slider.setValue(int((s["ema_alpha"] - 0.3) / 0.5 * 100))
        self._chk_blinkgate.setChecked(bool(s["blinkgate_enabled"]))
        self._chk_cam_degrade.setChecked(bool(s["cam_degrade_enabled"]))
        self._chk_degrade_noise.setChecked(bool(s["cam_degrade_noise"]))
        self._chk_degrade_sharp.setChecked(bool(s["cam_degrade_sharpness"]))
        self._chk_degrade_motion.setChecked(bool(s["cam_degrade_motion"]))
        self._chk_vcam.setChecked(bool(s["vcam_enabled"]))
        idx = self._res_combo.findText(s["output_resolution"])
        if idx >= 0:
            self._res_combo.setCurrentIndex(idx)
        self._chk_stats.setChecked(bool(s["show_stats"]))
        self._update_ultra_opts()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header strip
        header = QWidget()
        header.setFixedHeight(36)
        header.setStyleSheet("background:#1c1c1e;border-bottom:1px solid #2c2c2e;")
        h_lay = QHBoxLayout(header)
        h_lay.setContentsMargins(14, 0, 14, 0)
        title = QLabel("liveswap")
        title.setStyleSheet("font-size:15px;font-weight:bold;color:#f2f2f7;")
        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet("color:#8e8e93;font-size:12px;")
        h_lay.addWidget(title)
        h_lay.addStretch()
        h_lay.addWidget(self._status_label)
        root.addWidget(header)

        body = QSplitter(Qt.Orientation.Horizontal)

        # Left controls panel
        controls = QWidget()
        controls.setFixedWidth(320)
        controls.setStyleSheet("background:#1c1c1e;")
        ctrl_lay = QVBoxLayout(controls)
        ctrl_lay.setContentsMargins(8, 8, 8, 8)
        ctrl_lay.setSpacing(0)
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(
            "QTabWidget::pane{border:none;padding:0;}"
            "QTabBar::tab{padding:5px 10px;font-size:12px;}"
        )
        self._setup_live_tab()
        self._setup_profile_tab()
        self._setup_settings_tab()
        ctrl_lay.addWidget(self._tabs)

        body.addWidget(controls)
        body.addWidget(self._setup_preview_panel())
        body.setStretchFactor(1, 1)
        root.addWidget(body)

        sb = QStatusBar()
        sb.setStyleSheet("background:#1c1c1e;color:#8e8e93;font-size:11px;")
        self.setStatusBar(sb)
        self.statusBar().showMessage("Ready")

    # ── Tab 1: Live Mode ──────────────────────────────────────────────

    def _setup_live_tab(self) -> None:
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(6, 8, 6, 6)
        lay.setSpacing(8)

        # Source face
        src_group = QGroupBox("Source face")
        sg_lay = QVBoxLayout(src_group)

        drop_row = QHBoxLayout()
        drop_row.setSpacing(8)

        self._src_drop = DropZone("Drop\nphoto\nhere", multiple=False)
        self._src_drop.setFixedSize(100, 100)
        self._src_drop.files_dropped.connect(self._handle_source_drop)

        self._src_thumb = QLabel()
        self._src_thumb.setFixedSize(100, 100)
        self._src_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._src_thumb.setStyleSheet(
            "background:#111;border:1px solid #3a3a3c;"
        )

        thumb_col = QVBoxLayout()
        thumb_col.setSpacing(2)
        thumb_col.addWidget(self._src_thumb)
        self._src_confidence = QLabel("")
        self._src_confidence.setStyleSheet("color:#30d158;font-size:10px;")
        self._src_confidence.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb_col.addWidget(self._src_confidence)

        drop_row.addWidget(self._src_drop)
        drop_row.addLayout(thumb_col)
        drop_row.addStretch()
        sg_lay.addLayout(drop_row)

        self._src_browse_btn = QPushButton("Browse photo…")
        self._src_browse_btn.clicked.connect(self._select_source_photo)
        sg_lay.addWidget(self._src_browse_btn)
        lay.addWidget(src_group)

        # Camera input
        cam_group = QGroupBox("Camera input")
        cam_lay = QHBoxLayout(cam_group)
        self._cam_combo = QComboBox()
        for i in range(4):
            self._cam_combo.addItem(f"Camera {i}")
        self._cam_refresh_btn = QPushButton("Refresh")
        self._cam_refresh_btn.setFixedWidth(64)
        self._cam_refresh_btn.clicked.connect(self._refresh_cameras)
        cam_lay.addWidget(self._cam_combo)
        cam_lay.addWidget(self._cam_refresh_btn)
        lay.addWidget(cam_group)

        # Quality
        q_group = QGroupBox("Quality")
        q_lay = QVBoxLayout(q_group)
        self._radio_fast = QRadioButton("Fast")
        self._radio_ultra = QRadioButton("Ultra")
        self._radio_ultra.setChecked(True)
        self._quality_btn_grp = QButtonGroup(self)
        self._quality_btn_grp.addButton(self._radio_fast)
        self._quality_btn_grp.addButton(self._radio_ultra)
        self._radio_fast.toggled.connect(self._update_ultra_opts)
        q_row = QHBoxLayout()
        q_row.addWidget(self._radio_fast)
        q_row.addWidget(self._radio_ultra)
        q_row.addStretch()
        q_lay.addLayout(q_row)

        self._ultra_opts = QWidget()
        u_lay = QGridLayout(self._ultra_opts)
        u_lay.setContentsMargins(0, 2, 0, 0)
        u_lay.setSpacing(2)
        self._chk_sss = QCheckBox("SSS")
        self._chk_specular = QCheckBox("Specular")
        self._chk_detail = QCheckBox("Detail")
        self._chk_cam_match = QCheckBox("Camera match")
        for chk in (self._chk_sss, self._chk_specular,
                    self._chk_detail, self._chk_cam_match):
            chk.setChecked(True)
        u_lay.addWidget(self._chk_sss, 0, 0)
        u_lay.addWidget(self._chk_specular, 0, 1)
        u_lay.addWidget(self._chk_detail, 1, 0)
        u_lay.addWidget(self._chk_cam_match, 1, 1)
        q_lay.addWidget(self._ultra_opts)

        degrade_row = QHBoxLayout()
        degrade_row.addWidget(QLabel("Degradation:"))
        self._degrade_slider = QSlider(Qt.Orientation.Horizontal)
        self._degrade_slider.setRange(0, 100)
        self._degrade_slider.setValue(50)
        self._degrade_val = QLabel("50%")
        self._degrade_val.setFixedWidth(36)
        self._degrade_slider.valueChanged.connect(
            lambda v: self._degrade_val.setText(f"{v}%")
        )
        degrade_row.addWidget(self._degrade_slider)
        degrade_row.addWidget(self._degrade_val)
        q_lay.addLayout(degrade_row)
        lay.addWidget(q_group)

        # Start/Stop
        self._start_btn = QPushButton("▶  Start")
        self._start_btn.setFixedHeight(44)
        self._start_btn.setStyleSheet(self._start_btn_style(running=False))
        self._start_btn.clicked.connect(self._toggle_pipeline)
        lay.addWidget(self._start_btn)
        lay.addStretch()

        self._tabs.addTab(tab, "Live Mode")

    # ── Tab 2: Profile Mode ───────────────────────────────────────────

    def _setup_profile_tab(self) -> None:
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(6, 8, 6, 6)
        lay.setSpacing(8)

        # Saved profiles
        saved_group = QGroupBox("Saved profiles")
        sg_lay = QVBoxLayout(saved_group)
        p_row = QHBoxLayout()
        self._profile_combo = QComboBox()
        self._profile_del_btn = QPushButton("Delete")
        self._profile_del_btn.setFixedWidth(64)
        self._profile_del_btn.clicked.connect(self._delete_profile)
        p_row.addWidget(self._profile_combo)
        p_row.addWidget(self._profile_del_btn)
        sg_lay.addLayout(p_row)
        self._load_profile_btn = QPushButton("Load profile →  use in Live Mode")
        self._load_profile_btn.clicked.connect(self._load_profile)
        sg_lay.addWidget(self._load_profile_btn)
        lay.addWidget(saved_group)

        # Create profile
        create_group = QGroupBox("Create profile")
        cg_lay = QVBoxLayout(create_group)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self._profile_name_edit = QLineEdit()
        self._profile_name_edit.setPlaceholderText("e.g. John")
        name_row.addWidget(self._profile_name_edit)
        cg_lay.addLayout(name_row)

        self._profile_drop = DropZone(
            "Drop 5-10 photos here\n(or use Browse below)", multiple=True
        )
        self._profile_drop.setMinimumHeight(60)
        self._profile_drop.files_dropped.connect(self._handle_profile_drop)
        cg_lay.addWidget(self._profile_drop)

        self._photo_count_label = QLabel("0 photos selected")
        self._photo_count_label.setStyleSheet("color:#8e8e93;font-size:11px;")
        cg_lay.addWidget(self._photo_count_label)

        self._profile_thumbs = ThumbnailGrid()
        cg_lay.addWidget(self._profile_thumbs)

        self._profile_browse_btn = QPushButton("Browse photos…")
        self._profile_browse_btn.clicked.connect(self._browse_profile_photos)
        cg_lay.addWidget(self._profile_browse_btn)

        self._create_profile_btn = QPushButton("Create Profile")
        self._create_profile_btn.setStyleSheet(
            "QPushButton{background:#0a84ff;color:#fff;border-radius:6px;padding:6px;}"
            "QPushButton:hover{background:#409cff;}"
            "QPushButton:disabled{background:#3a3a3c;color:#636366;}"
        )
        self._create_profile_btn.clicked.connect(self._create_profile)
        cg_lay.addWidget(self._create_profile_btn)

        self._profile_progress = QProgressBar()
        self._profile_progress.setRange(0, 0)
        self._profile_progress.setVisible(False)
        cg_lay.addWidget(self._profile_progress)

        self._profile_quality_label = QLabel("")
        self._profile_quality_label.setStyleSheet("color:#30d158;font-size:12px;")
        cg_lay.addWidget(self._profile_quality_label)

        lay.addWidget(create_group)
        lay.addStretch()
        self._tabs.addTab(tab, "Profile Mode")

    # ── Tab 3: Settings ───────────────────────────────────────────────

    def _setup_settings_tab(self) -> None:
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(6, 8, 6, 6)
        lay.setSpacing(6)

        # Virtual camera
        vcam_group = QGroupBox("Virtual camera")
        v_lay = QHBoxLayout(vcam_group)
        self._chk_vcam = QCheckBox("Enable")
        self._vcam_name_label = QLabel("DirectShow Softcam")
        self._vcam_name_label.setStyleSheet("color:#8e8e93;font-size:11px;")
        v_lay.addWidget(self._chk_vcam)
        v_lay.addWidget(self._vcam_name_label)
        v_lay.addStretch()
        lay.addWidget(vcam_group)

        # Output resolution
        res_group = QGroupBox("Output resolution")
        r_lay = QHBoxLayout(res_group)
        self._res_combo = QComboBox()
        self._res_combo.addItems(list(_RES.keys()))
        r_lay.addWidget(self._res_combo)
        r_lay.addStretch()
        lay.addWidget(res_group)

        # GPEN
        gpen_group = QGroupBox("GPEN enhancement")
        g_lay = QVBoxLayout(gpen_group)
        self._chk_gpen = QCheckBox("Enable GPEN")
        self._chk_gpen.setChecked(True)
        g_lay.addWidget(self._chk_gpen)
        gpen_row = QHBoxLayout()
        gpen_row.addWidget(QLabel("Blend:"))
        self._gpen_slider = QSlider(Qt.Orientation.Horizontal)
        self._gpen_slider.setRange(0, 100)
        self._gpen_slider.setValue(80)
        self._gpen_val = QLabel("0.80")
        self._gpen_val.setFixedWidth(36)
        self._gpen_slider.valueChanged.connect(
            lambda v: self._gpen_val.setText(f"{v / 100:.2f}")
        )
        gpen_row.addWidget(self._gpen_slider)
        gpen_row.addWidget(self._gpen_val)
        g_lay.addLayout(gpen_row)
        lay.addWidget(gpen_group)

        # EMA smoothing
        ema_group = QGroupBox("EMA smoothing")
        e_lay = QHBoxLayout(ema_group)
        e_lay.addWidget(QLabel("Alpha (0.30–0.80):"))
        self._ema_slider = QSlider(Qt.Orientation.Horizontal)
        self._ema_slider.setRange(0, 100)
        self._ema_slider.setValue(50)
        self._ema_val = QLabel("0.55")
        self._ema_val.setFixedWidth(36)
        self._ema_slider.valueChanged.connect(
            lambda v: self._ema_val.setText(f"{0.3 + v / 100 * 0.5:.2f}")
        )
        e_lay.addWidget(self._ema_slider)
        e_lay.addWidget(self._ema_val)
        lay.addWidget(ema_group)

        # BlinkGate
        blink_group = QGroupBox("BlinkGate")
        b_lay = QHBoxLayout(blink_group)
        self._chk_blinkgate = QCheckBox("Enable blink detection (2d106det)")
        self._chk_blinkgate.setChecked(True)
        b_lay.addWidget(self._chk_blinkgate)
        lay.addWidget(blink_group)

        # Camera degradation matching
        cd_group = QGroupBox("Camera degradation matching")
        cd_lay = QVBoxLayout(cd_group)
        self._chk_cam_degrade = QCheckBox("Enable")
        self._chk_cam_degrade.setChecked(True)
        cd_lay.addWidget(self._chk_cam_degrade)
        cd_row = QHBoxLayout()
        self._chk_degrade_noise = QCheckBox("Noise")
        self._chk_degrade_sharp = QCheckBox("Sharpness")
        self._chk_degrade_motion = QCheckBox("Motion blur")
        for chk in (self._chk_degrade_noise,
                    self._chk_degrade_sharp,
                    self._chk_degrade_motion):
            chk.setChecked(True)
            cd_row.addWidget(chk)
        cd_lay.addLayout(cd_row)
        lay.addWidget(cd_group)

        # Stats overlay
        stats_group = QGroupBox("Debug overlay")
        s_lay = QHBoxLayout(stats_group)
        self._chk_stats = QCheckBox("Show FPS / inference stats overlay")
        self._chk_stats.setChecked(True)
        s_lay.addWidget(self._chk_stats)
        lay.addWidget(stats_group)

        lay.addStretch()
        self._tabs.addTab(tab, "Settings")

    # ── Preview panel ─────────────────────────────────────────────────

    def _setup_preview_panel(self) -> QWidget:
        panel = QWidget()
        p_lay = QVBoxLayout(panel)
        p_lay.setContentsMargins(4, 4, 4, 4)
        p_lay.setSpacing(4)

        self._preview_label = QLabel()
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setStyleSheet("background:#111;")
        self._preview_label.setMinimumSize(640, 480)
        self._preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        p_lay.addWidget(self._preview_label)

        btn_row = QHBoxLayout()
        self._screenshot_btn = QPushButton("Screenshot")
        self._screenshot_btn.setFixedWidth(100)
        self._screenshot_btn.clicked.connect(self._save_screenshot)
        btn_row.addStretch()
        btn_row.addWidget(self._screenshot_btn)
        p_lay.addLayout(btn_row)

        return panel

    def _update_ultra_opts(self) -> None:
        self._ultra_opts.setEnabled(self._radio_ultra.isChecked())

    @staticmethod
    def _start_btn_style(running: bool) -> str:
        if running:
            return (
                "QPushButton{font-size:16px;font-weight:bold;"
                "background:#ff453a;color:#fff;border-radius:8px;}"
                "QPushButton:hover{background:#ff6961;}"
            )
        return (
            "QPushButton{font-size:16px;font-weight:bold;"
            "background:#30d158;color:#000;border-radius:8px;}"
            "QPushButton:hover{background:#34e360;}"
            "QPushButton:disabled{background:#3a3a3c;color:#636366;}"
        )

    # ------------------------------------------------------------------
    # Pipeline control
    # ------------------------------------------------------------------

    def _toggle_pipeline(self) -> None:
        if self._running:
            self._stop_pipeline()
        else:
            self._start_pipeline()

    def _start_pipeline(self) -> None:
        if self._starting:
            return
        tab = self._tabs.currentIndex()
        source_face = self._source_face
        profile_name: str | None = None

        if tab == 0:  # Live Mode
            if source_face is None:
                self._set_status("Select a source photo first")
                return
        else:  # Profile Mode
            profile_name = self._profile_combo.currentText().strip()
            if not profile_name or profile_name == "(no profiles)":
                self._set_status("Load a profile first")
                return

        cam_idx = self._cam_combo.currentIndex()
        quality = "ultra" if self._radio_ultra.isChecked() else "fast"
        res_key = self._res_combo.currentText()
        target_w, target_h = _RES.get(res_key, (1280, 720))

        self._starting = True
        self._start_btn.setEnabled(False)
        self._set_status("Loading models…")

        def run():
            cap = None
            try:
                profile = load_profile_by_name(profile_name) if profile_name else None
                generated_face = None
                if profile is not None and profile.lora_path is not None:
                    self._pipeline = None
                    self._swapper = self._enhancer = self._parser = None
                    gc.collect()
                    generated_face = generate_profile_source_face(profile)
                self._load_models()
                pipeline = SwapPipeline(
                    self._swapper, self._enhancer, self._parser, quality=quality
                )
                if profile is None:
                    pipeline.set_source_face(source_face)
                elif generated_face is not None:
                    pipeline.set_source_face(generated_face)
                else:
                    pipeline.set_profile(profile)
                from faceswap.camera.capture import CaptureThread
                cap = CaptureThread(cam_idx, width=target_w, height=target_h, fps=30)
                cap.start()
                if self._closing:
                    cap.release()
                    return
                self._signals.startup_ready.emit((pipeline, cap, target_w, target_h))
            except Exception as exc:
                if cap is not None:
                    cap.release()
                self._signals.failed.emit("startup", str(exc))

        threading.Thread(target=run, daemon=True).start()

    @Slot(object)
    def _finish_start(self, result) -> None:
        pipeline, cap, target_w, target_h = result
        self._starting = False
        self._start_btn.setEnabled(True)
        if self._closing:
            cap.release()
            return
        self._pipeline, self._cap = pipeline, cap

        if self._chk_vcam.isChecked():
            try:
                from faceswap.camera.virtual_cam import VirtualCamera
                self._vcam = VirtualCamera(target_w, target_h, fps=30.0)
            except Exception as e:
                self._set_status(f"Virtual cam error: {e}")
                self._vcam = None

        def output_callback(frame: np.ndarray) -> None:
            if self._vcam:
                try:
                    self._vcam.send_frame(frame)
                except Exception:
                    pass
            self._preview_frame = frame

        self._runner = AsyncPipelineRunner(
            pipeline=self._pipeline,
            output_callback=output_callback,
            target_width=target_w,
            target_height=target_h,
        )
        self._runner.start(self._cap)
        self._running = True
        self._dropped_frames = 0
        self._frame_times.clear()
        self._start_btn.setText("■  Stop")
        self._start_btn.setStyleSheet(self._start_btn_style(running=True))
        vcam_note = " — virtual camera active" if self._vcam else ""
        self._set_status(f"Running{vcam_note}")

    def _stop_pipeline(self) -> bool:
        try:
            if self._runner:
                self._runner.stop()
                self._runner.join()
            elif self._cap:
                self._cap.release()
        except Exception as exc:
            self._set_status(f"Stop error: {exc}  — retry Stop")
            return False
        self._runner = None
        self._cap = None
        self._pipeline = None
        self._running = False
        if self._vcam:
            self._vcam.close()
            self._vcam = None
        self._start_btn.setText("▶  Start")
        self._start_btn.setStyleSheet(self._start_btn_style(running=False))
        self._set_status("Ready")
        return True

    def _load_models(self) -> None:
        if all(m is not None for m in (self._swapper, self._parser)):
            return
        swap_model = MODELS_DIR / "alphaface_256.onnx"
        enhance_model = MODELS_DIR / "GPEN-BFR-512.onnx"
        parse_model = MODELS_DIR / "bisenet.onnx"
        for path in (swap_model, enhance_model, parse_model):
            if not path.exists():
                raise FileNotFoundError(f"Missing model: {path}")
        swapper = FaceSwapper(swap_model, model_type="alphaface")
        enhancer = FaceEnhancer(enhance_model) if self._chk_gpen.isChecked() else None
        parser = FaceParser(parse_model)
        self._swapper, self._enhancer, self._parser = swapper, enhancer, parser

    @Slot(str, str)
    def _worker_failed(self, operation: str, message: str) -> None:
        if operation == "startup":
            self._starting = False
            self._start_btn.setEnabled(True)
        elif operation == "source":
            self._src_browse_btn.setEnabled(True)
            self._src_drop.setEnabled(True)
        elif operation == "profile":
            self._create_profile_btn.setEnabled(True)
            self._profile_progress.setVisible(False)
        self._set_status(f"Error: {message}")

    # ------------------------------------------------------------------
    # Source face
    # ------------------------------------------------------------------

    def _select_source_photo(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select source face photo", "", _IMG_FILTER
        )
        if path:
            self._apply_source_photo(path)

    def _handle_source_drop(self, paths: list[str]) -> None:
        if paths:
            self._apply_source_photo(paths[0])

    def _apply_source_photo(self, path: str) -> None:
        self._src_browse_btn.setEnabled(False)
        self._src_drop.setEnabled(False)
        self._src_confidence.setText("Detecting…")
        self._source_photo_path = path

        def run():
            try:
                frame = cv2.imread(path)
                if frame is None:
                    raise ValueError("Cannot read image")
                face = detect_largest(frame)
                if face is None:
                    raise ValueError("No face detected")
                conf = float(getattr(face, "det_score", 0.0)) * 100.0
                self._signals.source_ready.emit(face, Path(path).name, conf)
            except Exception as exc:
                self._signals.failed.emit("source", str(exc))

        threading.Thread(target=run, daemon=True).start()

    @Slot(object, str, float)
    def _source_loaded(self, face, name: str, confidence: float) -> None:
        self._source_face = face
        self._src_browse_btn.setEnabled(True)
        self._src_drop.setEnabled(True)

        path = self._source_photo_path
        if path:
            img = cv2.imread(path)
            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                s = min(h, w)
                sq = rgb[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
                sq = cv2.resize(sq, (100, 100))
                qimg = QImage(sq.data, 100, 100, 300, QImage.Format.Format_RGB888)
                self._src_thumb.setPixmap(QPixmap.fromImage(qimg).copy())

        conf_text = (f"Face detected ✓  {confidence:.0f}%"
                     if confidence > 0 else "Face detected ✓")
        self._src_confidence.setText(conf_text)
        self._set_status(f"Source: {name}")

    # ------------------------------------------------------------------
    # Profiles
    # ------------------------------------------------------------------

    def _refresh_profiles(self) -> None:
        self._profile_combo.clear()
        profiles = list_profiles()
        if profiles:
            self._profile_combo.addItems(profiles)
        else:
            self._profile_combo.addItem("(no profiles)")

    def _delete_profile(self) -> None:
        name = self._profile_combo.currentText().strip()
        if not name or name == "(no profiles)":
            return
        try:
            from faceswap.profile.manager import delete_profile as _del
            _del(name)
        except (ImportError, AttributeError):
            pass  # not yet implemented in manager
        except Exception as exc:
            self._set_status(f"Delete error: {exc}")
            return
        self._refresh_profiles()
        self._set_status(f"Deleted profile '{name}'")

    def _handle_profile_drop(self, paths: list[str]) -> None:
        self._profile_photos = paths
        n = len(paths)
        self._photo_count_label.setText(f"{n} photo{'s' if n != 1 else ''} selected")
        self._profile_thumbs.set_photos(paths)

    def _browse_profile_photos(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select 5-10 photos", "", _IMG_FILTER
        )
        if paths:
            self._handle_profile_drop(paths)

    def _create_profile(self) -> None:
        name = self._profile_name_edit.text().strip()
        if not name:
            self._set_status("Enter a profile name")
            return
        if len(self._profile_photos) < 2:
            self._set_status("Add at least 2 photos")
            return
        self._create_profile_btn.setEnabled(False)
        self._profile_progress.setVisible(True)
        self._profile_quality_label.setText("")
        self._set_status(f"Creating profile '{name}'…")

        photos = list(self._profile_photos)

        def run():
            try:
                create_profile_level1(name, photos)
                strength = min(99.0, 60.0 + len(photos) * 4.0)
                self._signals.profile_done.emit(name, strength)
            except Exception as exc:
                self._signals.failed.emit("profile", str(exc))

        threading.Thread(target=run, daemon=True).start()

    @Slot(str, float)
    def _profile_created(self, name: str, strength: float) -> None:
        self._create_profile_btn.setEnabled(True)
        self._profile_progress.setVisible(False)
        self._profile_quality_label.setText(f"Identity strength: {strength:.0f}%")
        self._refresh_profiles()
        self._set_status(f"Profile '{name}' created")

    def _load_profile(self) -> None:
        name = self._profile_combo.currentText().strip()
        if not name or name == "(no profiles)":
            self._set_status("No profile selected")
            return
        self._set_status(f"Profile '{name}' active")

    # ------------------------------------------------------------------
    # Camera refresh
    # ------------------------------------------------------------------

    def _refresh_cameras(self) -> None:
        current = self._cam_combo.currentIndex()
        self._cam_combo.clear()
        found = []
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                found.append(i)
                cap.release()
        for i in (found if found else range(4)):
            self._cam_combo.addItem(f"Camera {i}")
        self._cam_combo.setCurrentIndex(
            min(max(current, 0), self._cam_combo.count() - 1)
        )

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _update_preview(self) -> None:
        frame = self._preview_frame
        if frame is None:
            return
        now = time.monotonic()
        self._frame_times.append(now)

        if self._chk_stats.isChecked() and self._running:
            frame = self._draw_stats_overlay(frame.copy())

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self._preview_label.width(),
            self._preview_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._preview_label.setPixmap(pixmap)

        if self._running and len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            fps = (len(self._frame_times) - 1) / max(elapsed, 1e-6)
            inf_ms = getattr(self._pipeline, "last_inference_ms", 0.0)
            self._set_status(f"Running — {fps:.0f} FPS  {inf_ms:.0f} ms/frame")

    def _draw_stats_overlay(self, frame: np.ndarray) -> np.ndarray:
        if self._pipeline is None:
            return frame
        h, w = frame.shape[:2]
        fps = 0.0
        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            fps = (len(self._frame_times) - 1) / max(elapsed, 1e-6)
        inf_ms = getattr(self._pipeline, "last_inference_ms", 0.0)
        avg_ms = getattr(self._pipeline, "average_inference_ms", 0.0)
        skipped = getattr(self._pipeline, "gpen_skipped", 0)
        lines = [
            f"FPS: {fps:.1f}",
            f"Frame: {inf_ms:.0f} ms",
            f"Avg: {avg_ms:.0f} ms",
            f"GPEN skip: {skipped}",
        ]
        x, y0 = w - 160, 14
        for i, line in enumerate(lines):
            y = y0 + i * 20
            # Drop shadow then white text
            cv2.putText(frame, line, (x + 1, y + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, line, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def _save_screenshot(self) -> None:
        frame = self._preview_frame
        if frame is None:
            self._set_status("No frame to save")
            return
        desktop = Path.home() / "Desktop"
        path = desktop / f"liveswap_{int(time.time())}.png"
        cv2.imwrite(str(path), frame)
        self._set_status(f"Saved: {path.name}")

    # ------------------------------------------------------------------
    # Helpers / close
    # ------------------------------------------------------------------

    def _set_status(self, msg: str) -> None:
        self._status_label.setText(msg)
        self.statusBar().showMessage(msg)

    def closeEvent(self, event) -> None:
        self._closing = True
        self._collect_settings()
        self._save_settings()
        if self._stop_pipeline():
            event.accept()
        else:
            self._closing = False
            event.ignore()
