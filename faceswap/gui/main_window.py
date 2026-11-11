"""
Main application window (PySide6).

Two tabs:
  - Live Mode:    source photo/webcam → instant swap
  - Profile Mode: upload photos → create profile → high quality swap

Controls:
  - Camera selector
  - Source selector (live mode: photo or webcam)
  - Profile selector (profile mode)
  - Quality toggle (fast / ultra)
  - Virtual camera output toggle
  - Preview
"""
from __future__ import annotations
import gc
import threading
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal, QObject, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QTabWidget,
    QFileDialog, QProgressBar, QCheckBox, QSlider,
    QGroupBox, QStatusBar, QSplitter,
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


class WorkerSignals(QObject):
    startup_ready = Signal(object)
    failed = Signal(str, str)
    source_ready = Signal(object, str)
    profile_done = Signal(str)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("liveswap")
        self.setMinimumSize(1200, 700)

        self._pipeline: SwapPipeline | None = None
        self._runner: AsyncPipelineRunner | None = None
        self._cap = None
        self._source_face = None
        self._swapper = self._enhancer = self._parser = None
        self._starting = False
        self._closing = False
        self._signals = WorkerSignals(self)
        self._signals.startup_ready.connect(self._finish_start)
        self._signals.failed.connect(self._worker_failed)
        self._signals.source_ready.connect(self._source_loaded)
        self._signals.profile_done.connect(self._profile_created)
        self._vcam = None
        self._running = False

        self._setup_ui()
        self._refresh_profiles()

        # Preview timer
        self._preview_frame: np.ndarray | None = None
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_preview)
        self._timer.start(33)  # ~30fps

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left: controls
        controls = QWidget()
        controls.setFixedWidth(280)
        ctrl_layout = QVBoxLayout(controls)

        # Camera selector
        cam_group = QGroupBox("Camera Input")
        cam_layout = QVBoxLayout(cam_group)
        self._cam_combo = QComboBox()
        self._cam_combo.addItems(["Camera 0", "Camera 1", "Camera 2"])
        cam_layout.addWidget(self._cam_combo)
        ctrl_layout.addWidget(cam_group)

        # Mode tabs
        self._tabs = QTabWidget()

        # Live mode tab
        live_tab = QWidget()
        live_layout = QVBoxLayout(live_tab)
        self._source_photo_btn = QPushButton("Select Source Photo")
        self._source_photo_btn.clicked.connect(self._select_source_photo)
        self._source_photo_label = QLabel("No photo selected")
        self._source_photo_label.setWordWrap(True)
        live_layout.addWidget(self._source_photo_btn)
        live_layout.addWidget(self._source_photo_label)
        live_layout.addStretch()
        self._tabs.addTab(live_tab, "Live Mode")

        # Profile mode tab
        profile_tab = QWidget()
        prof_layout = QVBoxLayout(profile_tab)
        self._profile_combo = QComboBox()
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self._refresh_profiles)
        self._create_profile_btn = QPushButton("New Profile (2 min)")
        self._create_profile_btn.clicked.connect(self._create_profile)
        self._profile_progress = QProgressBar()
        self._profile_progress.setVisible(False)
        prof_layout.addWidget(QLabel("Select Profile:"))
        prof_layout.addWidget(self._profile_combo)
        prof_layout.addWidget(self._refresh_btn)
        prof_layout.addWidget(self._create_profile_btn)
        prof_layout.addWidget(self._profile_progress)
        prof_layout.addStretch()
        self._tabs.addTab(profile_tab, "Profile Mode")

        ctrl_layout.addWidget(self._tabs)

        # Quality selector
        quality_group = QGroupBox("Quality")
        q_layout = QVBoxLayout(quality_group)
        self._quality_ultra = QCheckBox("Ultra quality (SSS + specular + detail)")
        self._quality_ultra.setChecked(True)
        q_layout.addWidget(self._quality_ultra)
        ctrl_layout.addWidget(quality_group)

        # Virtual camera
        vcam_group = QGroupBox("Output")
        vcam_layout = QVBoxLayout(vcam_group)
        self._vcam_check = QCheckBox("Virtual Camera (DirectShow Softcam)")
        vcam_layout.addWidget(self._vcam_check)
        self._vcam_name_label = QLabel("Device: 'DirectShow Softcam'")
        vcam_layout.addWidget(self._vcam_name_label)
        ctrl_layout.addWidget(vcam_group)

        # Start/Stop button
        self._start_btn = QPushButton("▶  Start")
        self._start_btn.setFixedHeight(44)
        self._start_btn.setStyleSheet("font-size: 16px; font-weight: bold;")
        self._start_btn.clicked.connect(self._toggle_pipeline)
        ctrl_layout.addWidget(self._start_btn)
        ctrl_layout.addStretch()

        # Right: preview
        preview_widget = QWidget()
        prev_layout = QVBoxLayout(preview_widget)
        self._preview_label = QLabel()
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setStyleSheet("background: #111;")
        self._preview_label.setMinimumSize(640, 480)
        prev_layout.addWidget(self._preview_label)
        prev_layout.addWidget(QLabel("Preview — virtual camera output"))

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(controls)
        splitter.addWidget(preview_widget)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")

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
        profile_name = self._profile_combo.currentText() if self._tabs.currentIndex() else None
        if profile_name is not None and (
            not self._profile_combo.count() or not profile_name.strip()
            or profile_name == "(no profiles)"
        ):
            self.statusBar().showMessage("Create a profile first")
            return
        if profile_name is None and self._source_face is None:
            self.statusBar().showMessage("Select a source photo first")
            return
        cam_idx = self._cam_combo.currentIndex()
        quality = "ultra" if self._quality_ultra.isChecked() else "fast"
        source_face = self._source_face
        self._starting = True
        self._start_btn.setEnabled(False)
        self.statusBar().showMessage("Loading models and source...")

        def run():
            cap = None
            try:
                profile = load_profile_by_name(profile_name) if profile_name is not None else None
                generated_face = None
                if profile is not None and profile.lora_path is not None:
                    # Diffusion needs the GPU memory occupied by cached ONNX sessions.
                    self._pipeline = None
                    self._swapper = self._enhancer = self._parser = None
                    gc.collect()
                    generated_face = generate_profile_source_face(profile)
                self._load_models()
                pipeline = SwapPipeline(self._swapper, self._enhancer, self._parser,
                                        quality=quality)
                if profile is None:
                    pipeline.set_source_face(source_face)
                elif generated_face is not None:
                    pipeline.set_source_face(generated_face)
                else:
                    pipeline.set_profile(profile)
                from faceswap.camera.capture import CaptureThread
                cap = CaptureThread(cam_idx, width=1280, height=720, fps=30)
                cap.start()
                if self._closing:
                    cap.release()
                    return
                self._signals.startup_ready.emit((pipeline, cap))
            except Exception as exc:
                if cap is not None:
                    cap.release()
                self._signals.failed.emit("startup", str(exc))

        threading.Thread(target=run, daemon=True).start()

    @Slot(object)
    def _finish_start(self, result) -> None:
        pipeline, cap = result
        self._starting = False
        self._start_btn.setEnabled(True)
        if self._closing:
            cap.release()
            return
        self._pipeline, self._cap = pipeline, cap

        # Virtual camera
        if self._vcam_check.isChecked():
            try:
                from faceswap.camera.virtual_cam import VirtualCamera
                self._vcam = VirtualCamera(1280, 720, fps=30.0)
            except Exception as e:
                self.statusBar().showMessage(f"Virtual cam error: {e}")
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
            target_width=1280,
            target_height=720,
        )
        self._runner.start(self._cap)
        self._running = True
        self._start_btn.setText("■  Stop")
        self.statusBar().showMessage("Running — virtual camera active" if self._vcam else "Running")

    def _stop_pipeline(self) -> bool:
        try:
            if self._runner:
                self._runner.stop()
                self._runner.join()
            elif self._cap:
                self._cap.release()
        except Exception as exc:
            # Keep the output alive while a worker may still be inside send_frame.
            self.statusBar().showMessage(f"Could not stop pipeline: {exc}; retry Stop")
            return False
        self._runner = None
        self._cap = None
        self._pipeline = None
        self._running = False
        if self._vcam:
            self._vcam.close()
            self._vcam = None
        self._start_btn.setText("Start")
        self.statusBar().showMessage("Stopped")
        return True

    def _load_models(self) -> None:
        if all(model is not None for model in (self._swapper, self._enhancer, self._parser)):
            return
        try:
            swap_model = MODELS_DIR / "alphaface_256.onnx"
            enhance_model = MODELS_DIR / "GPEN-BFR-512.onnx"
            parse_model = MODELS_DIR / "bisenet.onnx"
            for path in (swap_model, enhance_model, parse_model):
                if not path.exists():
                    raise FileNotFoundError(f"Missing model: {path}")
            swapper = FaceSwapper(swap_model, model_type="alphaface")
            enhancer = FaceEnhancer(enhance_model)
            parser = FaceParser(parse_model)
            self._swapper, self._enhancer, self._parser = swapper, enhancer, parser
        except Exception:
            self._swapper = self._enhancer = self._parser = None
            raise

    @Slot(str, str)
    def _worker_failed(self, operation: str, message: str) -> None:
        if operation == "startup":
            self._starting = False
            self._start_btn.setEnabled(True)
        elif operation == "source":
            self._source_photo_btn.setEnabled(True)
        elif operation == "profile":
            self._create_profile_btn.setEnabled(True)
            self._profile_progress.setVisible(False)
        self.statusBar().showMessage(message)

    # ------------------------------------------------------------------
    # Source / profile
    # ------------------------------------------------------------------

    def _select_source_photo(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Source Face Photo", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if not path:
            return

        self._source_photo_btn.setEnabled(False)
        self.statusBar().showMessage("Loading source face...")

        def run():
            try:
                frame = cv2.imread(path)
                if frame is None:
                    raise ValueError("Cannot read image")
                face = detect_largest(frame)
                if face is None:
                    raise ValueError("No face detected in photo")
                self._signals.source_ready.emit(face, Path(path).name)
            except Exception as exc:
                self._signals.failed.emit("source", str(exc))

        threading.Thread(target=run, daemon=True).start()

    @Slot(object, str)
    def _source_loaded(self, face, name: str) -> None:
        self._source_face = face
        self._source_photo_btn.setEnabled(True)
        self._source_photo_label.setText(name)
        self.statusBar().showMessage("Source face loaded; applied on next start")

    def _refresh_profiles(self) -> None:
        self._profile_combo.clear()
        profiles = list_profiles()
        self._profile_combo.addItems(profiles)
        self._profile_combo.setPlaceholderText("No profiles")

    def _create_profile(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select 5-10 photos of source person", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if len(paths) < 2:
            self.statusBar().showMessage("Select at least 2 photos")
            return

        name, ok = self._prompt_name()
        if not ok or not name:
            return

        self._create_profile_btn.setEnabled(False)
        self._profile_progress.setVisible(True)
        self._profile_progress.setValue(0)
        self.statusBar().showMessage("Creating profile...")

        def run():
            try:
                create_profile_level1(name, paths)
                self._signals.profile_done.emit(name)
            except Exception as e:
                self._signals.failed.emit("profile", f"Profile error: {e}")

        threading.Thread(target=run, daemon=True).start()

    @Slot(str)
    def _profile_created(self, name: str) -> None:
        self._refresh_profiles()
        self._create_profile_btn.setEnabled(True)
        self._profile_progress.setVisible(False)
        self.statusBar().showMessage(f"Profile '{name}' created")

    def _prompt_name(self) -> tuple[str, bool]:
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Profile Name", "Enter a name:")
        return name.strip(), ok

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _update_preview(self) -> None:
        frame = self._preview_frame
        if frame is None:
            return

        h, w, ch = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * ch, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(
            self._preview_label.width(),
            self._preview_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._preview_label.setPixmap(pixmap)

    def closeEvent(self, event) -> None:
        self._closing = True
        if self._stop_pipeline():
            event.accept()
        else:
            self._closing = False
            event.ignore()
