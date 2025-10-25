import sys
import cv2
import threading
import time
from ultralytics import YOLO
from PyQt6.QtCore import QTimer, Qt, pyqtSignal, QObject
from PyQt6.QtWidgets import (
    QApplication, QLabel, QVBoxLayout, QWidget, QPushButton, QHBoxLayout,
    QComboBox, QSlider, QGridLayout, QCheckBox, QGroupBox, QSpinBox,
    QTabWidget, QTextEdit, QProgressBar, QMessageBox, QFileDialog
)
from PyQt6.QtGui import QImage, QPixmap, QFont, QPalette, QColor
import os
import numpy as np
from collections import deque, defaultdict
from datetime import datetime
import json
import subprocess
import platform

# ----------------------------
# YOLO Models Configuration
# ----------------------------
YOLO_MODELS = {
    "YOLOv8n (Fastest)": "yolov8n.pt",
    "YOLOv8s": "yolov8s.pt",
    "YOLOv8m (Balanced)": "yolov8m.pt",
    "YOLOv8l": "yolov8l.pt",
    "YOLOv8x (Most Accurate)": "yolov8x.pt"
}
current_model_name = "YOLOv8m (Balanced)"
model = None

# ----------------------------
# Utility Functions
# ----------------------------
def set_camera_resolution(cap, width=1920, height=1080):
    """Set camera resolution and return actual values"""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution set to: {actual_width}x{actual_height}")
    return actual_width, actual_height

def get_available_cameras():
    """Detect all available cameras including Continuity Camera"""
    cameras = []
    
    if sys.platform == "darwin":  # macOS
        # Method 1: Try AVFoundation device enumeration
        try:
            # Use system_profiler to get camera info
            result = subprocess.run(
                ['system_profiler', 'SPCameraDataType'],
                capture_output=True,
                text=True,
                timeout=5
            )
            output = result.stdout
            
            # Parse output for camera names
            if "Camera" in output or "iPhone" in output or "iPad" in output:
                print("Detected cameras via system_profiler")
        except:
            pass
        
        # Method 2: Try numbered indices with AVFoundation
        print("Scanning for cameras...")
        for i in range(15):  # Scan more indices for Continuity Camera
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
                if cap.isOpened():
                    # Try to read a frame to verify camera works
                    ret, frame = cap.read()
                    if ret:
                        # Get camera name if possible
                        backend_name = cap.getBackendName()
                        cameras.append((f"Camera {i} (AVFoundation)", i))
                        print(f"Found working camera at index {i}")
                    cap.release()
            except Exception as e:
                print(f"Error checking camera {i}: {e}")
        
        # Add manual Continuity Camera option
        if len(cameras) > 1:
            # If multiple cameras found, one might be Continuity Camera
            cameras.append(("iPhone/iPad (Continuity - Try All)", "continuity"))
    else:
        # Windows/Linux
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cameras.append((f"Camera {i}", i))
                cap.release()
    
    if not cameras:
        cameras.append(("Default Camera (0)", 0))
    
    print(f"Total cameras found: {len(cameras)}")
    for name, idx in cameras:
        print(f"  - {name}: {idx}")
    
    return cameras

def draw_fancy_box(img, box, label, color, score=None):
    """Draw fancy styled bounding box with rounded corners"""
    x1, y1, x2, y2 = box
    
    # Draw corner brackets for modern look
    corner_length = 20
    thickness = 3
    
    # Top-left corner
    cv2.line(img, (x1, y1), (x1 + corner_length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + corner_length), color, thickness)
    
    # Top-right corner
    cv2.line(img, (x2, y1), (x2 - corner_length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + corner_length), color, thickness)
    
    # Bottom-left corner
    cv2.line(img, (x1, y2), (x1 + corner_length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - corner_length), color, thickness)
    
    # Bottom-right corner
    cv2.line(img, (x2, y2), (x2 - corner_length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - corner_length), color, thickness)
    
    # Draw thin connecting lines
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    
    return img

# ----------------------------
# Signal Emitter for Thread-Safe Updates
# ----------------------------
class SignalEmitter(QObject):
    update_stats = pyqtSignal(dict)
    update_log = pyqtSignal(str)
    show_message = pyqtSignal(str, str)  # title, message

# ----------------------------
# Main GUI Class
# ----------------------------
class VideoCaptureWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚀 YOLOv8 Pro Max - AI Vision Studio")
        self.setGeometry(50, 50, 1600, 950)
        
        # Set professional dark theme
        self.setStyleSheet("""
            QWidget {
                background-color: #1a1a2e;
                color: #eaeaea;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton {
                background-color: #16213e;
                color: white;
                border: 2px solid #0f3460;
                padding: 10px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #0f3460;
                border: 2px solid #e94560;
            }
            QPushButton:pressed {
                background-color: #0a2647;
            }
            QPushButton:disabled {
                background-color: #2a2a3e;
                color: #666;
                border: 2px solid #333;
            }
            QComboBox, QSpinBox {
                background-color: #16213e;
                color: white;
                border: 2px solid #0f3460;
                padding: 6px;
                border-radius: 4px;
                font-size: 10px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #e94560;
                margin-right: 5px;
            }
            QSlider::groove:horizontal {
                background: #0f3460;
                height: 10px;
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #e94560, stop:1 #f39c12);
                width: 20px;
                margin: -5px 0;
                border-radius: 10px;
            }
            QSlider::sub-page:horizontal {
                background: #e94560;
                border-radius: 5px;
            }
            QGroupBox {
                border: 2px solid #e94560;
                border-radius: 8px;
                margin-top: 12px;
                font-weight: bold;
                padding-top: 10px;
                font-size: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
                color: #e94560;
            }
            QTextEdit {
                background-color: #0f0f1e;
                color: #00ff41;
                border: 2px solid #0f3460;
                font-family: 'Courier New', monospace;
                border-radius: 4px;
                padding: 5px;
                font-size: 10px;
            }
            QCheckBox {
                spacing: 8px;
                font-size: 10px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 2px solid #0f3460;
                background-color: #16213e;
            }
            QCheckBox::indicator:checked {
                background-color: #e94560;
                border: 2px solid #e94560;
            }
            QLabel {
                font-size: 11px;
            }
            QProgressBar {
                border: 2px solid #0f3460;
                border-radius: 5px;
                text-align: center;
                background-color: #16213e;
                color: white;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #e94560, stop:1 #f39c12);
                border-radius: 3px;
            }
        """)

        # Initialize variables
        self.camera_index = 0
        self.cap = None
        self.width = 0
        self.height = 0
        self.signal_emitter = SignalEmitter()
        
        # FPS tracking with moving average
        self.fps_history = deque(maxlen=30)
        
        # Object tracking and analytics
        self.tracked_objects = {}
        self.object_history = deque(maxlen=1000)
        self.detection_history = defaultdict(list)
        self.session_start_time = time.time()
        
        # Time-lapse variables
        self.timelapse_active = False
        self.timelapse_interval = 2  # seconds
        self.timelapse_last_capture = 0
        self.timelapse_frames = []
        self.timelapse_counter = 0
        
        # Motion detection
        self.motion_detection = False
        self.last_frame_gray = None
        self.motion_threshold = 25
        
        # Alert system
        self.alert_objects = set()
        self.alert_sound_enabled = False
        
        # Initialize YOLO model
        global model
        model = YOLO(YOLO_MODELS[current_model_name])
        
        # Create UI
        self.init_ui()
        
        # Threading & processing
        self.lock = threading.Lock()
        self.frame = None
        self.results = None
        self.last_time = time.time()
        self.fps = 0
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.running = True
        self.recording = False
        self.video_writer = None
        self.processing_time = 0
        
        # Advanced features flags
        self.show_boxes = True
        self.show_labels = True
        self.show_confidence = True
        self.show_trails = False
        self.apply_blur = False
        self.track_objects = True
        self.fancy_boxes = True
        self.show_fps_overlay = True
        self.show_object_count = True
        
        # Initialize camera
        available_cameras = get_available_cameras()
        if available_cameras:
            self.init_camera(available_cameras[0][1])
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        self.capture_thread.start()
        
        # Timer for GUI update
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~33 FPS
        
        # Analytics timer (every 5 seconds)
        self.analytics_timer = QTimer()
        self.analytics_timer.timeout.connect(self.update_analytics)
        self.analytics_timer.start(5000)
        
        # Connect signals
        self.signal_emitter.update_log.connect(self.append_log)
        self.signal_emitter.show_message.connect(self.show_message_box)
        
        # Welcome message
        self.append_log("🚀 YOLOv8 Pro Max initialized successfully!")
        self.append_log(f"📱 Platform: {platform.system()} {platform.machine()}")
        self.append_log(f"🎯 Default model: {current_model_name}")

    def init_ui(self):
        """Initialize the user interface"""
        # Main layout
        main_layout = QHBoxLayout()
        
        # Left panel - Video feed
        left_panel = QVBoxLayout()
        
        # Video display
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(900, 650)
        self.image_label.setStyleSheet("""
            border: 3px solid #e94560; 
            background-color: #0f0f1e;
            border-radius: 8px;
        """)
        
        # Error/status label
        self.camera_error_label = QLabel("", self)
        self.camera_error_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.camera_error_label.setStyleSheet("color: #ff6b6b; background-color: transparent;")
        self.camera_error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Quick stats bar
        quick_stats_layout = QHBoxLayout()
        self.quick_fps = QLabel("⚡ FPS: 0")
        self.quick_objects = QLabel("🎯 Objects: 0")
        self.quick_model = QLabel(f"🤖 Model: {current_model_name}")
        
        for label in [self.quick_fps, self.quick_objects, self.quick_model]:
            label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            label.setStyleSheet("""
                background-color: #16213e; 
                padding: 8px; 
                border-radius: 5px;
                border: 1px solid #0f3460;
            """)
            quick_stats_layout.addWidget(label)
        
        left_panel.addWidget(self.image_label)
        left_panel.addWidget(self.camera_error_label)
        left_panel.addLayout(quick_stats_layout)
        
        # Right panel - Controls (with tabs)
        right_panel = QVBoxLayout()
        
        # Create tabbed interface
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #0f3460;
                border-radius: 5px;
                background-color: #1a1a2e;
            }
            QTabBar::tab {
                background-color: #16213e;
                color: white;
                padding: 8px 15px;
                margin-right: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #e94560;
            }
            QTabBar::tab:hover {
                background-color: #0f3460;
            }
        """)
        
        # Tab 1: Camera & Detection
        tab1 = QWidget()
        tab1_layout = QVBoxLayout()
        
        # Camera Controls
        camera_group = self.create_camera_controls()
        tab1_layout.addWidget(camera_group)
        
        # Detection Settings
        detection_group = self.create_detection_controls()
        tab1_layout.addWidget(detection_group)
        
        tab1_layout.addStretch()
        tab1.setLayout(tab1_layout)
        
        # Tab 2: Visual & Display
        tab2 = QWidget()
        tab2_layout = QVBoxLayout()
        
        display_group = self.create_display_controls()
        tab2_layout.addWidget(display_group)
        
        visual_effects_group = self.create_visual_effects_controls()
        tab2_layout.addWidget(visual_effects_group)
        
        tab2_layout.addStretch()
        tab2.setLayout(tab2_layout)
        
        # Tab 3: Recording & Capture
        tab3 = QWidget()
        tab3_layout = QVBoxLayout()
        
        recording_group = self.create_recording_controls()
        tab3_layout.addWidget(recording_group)
        
        timelapse_group = self.create_timelapse_controls()
        tab3_layout.addWidget(timelapse_group)
        
        tab3_layout.addStretch()
        tab3.setLayout(tab3_layout)
        
        # Tab 4: Advanced Features
        tab4 = QWidget()
        tab4_layout = QVBoxLayout()
        
        motion_group = self.create_motion_controls()
        tab4_layout.addWidget(motion_group)
        
        alert_group = self.create_alert_controls()
        tab4_layout.addWidget(alert_group)
        
        export_group = self.create_export_controls()
        tab4_layout.addWidget(export_group)
        
        tab4_layout.addStretch()
        tab4.setLayout(tab4_layout)
        
        # Add tabs
        self.tab_widget.addTab(tab1, "📹 Camera")
        self.tab_widget.addTab(tab2, "🎨 Display")
        self.tab_widget.addTab(tab3, "🎬 Record")
        self.tab_widget.addTab(tab4, "⚡ Advanced")
        
        right_panel.addWidget(self.tab_widget)
        
        # Statistics Display
        stats_group = self.create_statistics_display()
        right_panel.addWidget(stats_group)
        
        # Activity Log
        log_group = QGroupBox("📋 Activity Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        log_layout.addWidget(self.log_text)
        
        # Log controls
        log_controls = QHBoxLayout()
        clear_log_btn = QPushButton("Clear Log")
        clear_log_btn.clicked.connect(lambda: self.log_text.clear())
        export_log_btn = QPushButton("Export Log")
        export_log_btn.clicked.connect(self.export_log)
        log_controls.addWidget(clear_log_btn)
        log_controls.addWidget(export_log_btn)
        log_layout.addLayout(log_controls)
        
        log_group.setLayout(log_layout)
        right_panel.addWidget(log_group)
        
        # Add panels to main layout
        main_layout.addLayout(left_panel, 65)
        main_layout.addLayout(right_panel, 35)
        
        self.setLayout(main_layout)

    def create_camera_controls(self):
        """Create camera control group"""
        camera_group = QGroupBox("📹 Camera Settings")
        camera_layout = QGridLayout()
        
        camera_layout.addWidget(QLabel("Camera Source:"), 0, 0)
        self.camera_combo = QComboBox()
        available_cameras = get_available_cameras()
        for name, index in available_cameras:
            self.camera_combo.addItem(name, index)
        self.camera_combo.currentIndexChanged.connect(self.switch_camera)
        camera_layout.addWidget(self.camera_combo, 0, 1, 1, 2)
        
        refresh_cameras_btn = QPushButton("🔄 Refresh")
        refresh_cameras_btn.clicked.connect(self.refresh_cameras)
        camera_layout.addWidget(refresh_cameras_btn, 0, 3)
        
        camera_layout.addWidget(QLabel("Resolution:"), 1, 0)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["1920x1080 (FHD)", "1280x720 (HD)", "640x480 (SD)"])
        self.resolution_combo.setCurrentIndex(1)
        self.resolution_combo.currentTextChanged.connect(self.change_resolution)
        camera_layout.addWidget(self.resolution_combo, 1, 1, 1, 3)
        
        camera_layout.addWidget(QLabel("FPS Target:"), 2, 0)
        self.fps_target_spin = QSpinBox()
        self.fps_target_spin.setRange(10, 60)
        self.fps_target_spin.setValue(30)
        self.fps_target_spin.setSuffix(" fps")
        camera_layout.addWidget(self.fps_target_spin, 2, 1, 1, 3)
        
        camera_group.setLayout(camera_layout)
        return camera_group

    def create_detection_controls(self):
        """Create detection control group"""
        detection_group = QGroupBox("🎯 Detection Settings")
        detection_layout = QGridLayout()
        
        detection_layout.addWidget(QLabel("YOLO Model:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(YOLO_MODELS.keys())
        self.model_combo.setCurrentText(current_model_name)
        self.model_combo.currentTextChanged.connect(self.switch_model)
        detection_layout.addWidget(self.model_combo, 0, 1, 1, 3)
        
        detection_layout.addWidget(QLabel("Confidence:"), 1, 0)
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setMinimum(1)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(25)
        self.conf_slider.valueChanged.connect(self.update_confidence)
        detection_layout.addWidget(self.conf_slider, 1, 1, 1, 2)
        self.conf_label = QLabel("0.25")
        self.conf_label.setStyleSheet("color: #e94560; font-weight: bold;")
        detection_layout.addWidget(self.conf_label, 1, 3)
        
        detection_layout.addWidget(QLabel("IOU Threshold:"), 2, 0)
        self.iou_slider = QSlider(Qt.Orientation.Horizontal)
        self.iou_slider.setMinimum(1)
        self.iou_slider.setMaximum(100)
        self.iou_slider.setValue(45)
        self.iou_slider.valueChanged.connect(self.update_iou)
        detection_layout.addWidget(self.iou_slider, 2, 1, 1, 2)
        self.iou_label = QLabel("0.45")
        self.iou_label.setStyleSheet("color: #e94560; font-weight: bold;")
        detection_layout.addWidget(self.iou_label, 2, 3)
        
        detection_group.setLayout(detection_layout)
        return detection_group

    def create_display_controls(self):
        """Create display control group"""
        display_group = QGroupBox("🎨 Display Options")
        display_layout = QVBoxLayout()
        
        self.box_checkbox = QCheckBox("Show Bounding Boxes")
        self.box_checkbox.setChecked(True)
        self.box_checkbox.stateChanged.connect(lambda: setattr(self, 'show_boxes', self.box_checkbox.isChecked()))
        
        self.label_checkbox = QCheckBox("Show Labels")
        self.label_checkbox.setChecked(True)
        self.label_checkbox.stateChanged.connect(lambda: setattr(self, 'show_labels', self.label_checkbox.isChecked()))
        
        self.conf_display_checkbox = QCheckBox("Show Confidence Scores")
        self.conf_display_checkbox.setChecked(True)
        self.conf_display_checkbox.stateChanged.connect(lambda: setattr(self, 'show_confidence', self.conf_display_checkbox.isChecked()))
        
        self.track_checkbox = QCheckBox("Enable Object Tracking")
        self.track_checkbox.setChecked(True)
        self.track_checkbox.stateChanged.connect(lambda: setattr(self, 'track_objects', self.track_checkbox.isChecked()))
        
        self.fancy_box_checkbox = QCheckBox("Fancy Box Style (Corners)")
        self.fancy_box_checkbox.setChecked(True)
        self.fancy_box_checkbox.stateChanged.connect(lambda: setattr(self, 'fancy_boxes', self.fancy_box_checkbox.isChecked()))
        
        self.fps_overlay_checkbox = QCheckBox("Show FPS Overlay on Video")
        self.fps_overlay_checkbox.setChecked(True)
        self.fps_overlay_checkbox.stateChanged.connect(lambda: setattr(self, 'show_fps_overlay', self.fps_overlay_checkbox.isChecked()))
        
        for checkbox in [self.box_checkbox, self.label_checkbox, self.conf_display_checkbox, 
                        self.track_checkbox, self.fancy_box_checkbox, self.fps_overlay_checkbox]:
            display_layout.addWidget(checkbox)
        
        display_group.setLayout(display_layout)
        return display_group

    def create_visual_effects_controls(self):
        """Create visual effects control group"""
        effects_group = QGroupBox("✨ Visual Effects")
        effects_layout = QVBoxLayout()
        
        self.blur_checkbox = QCheckBox("Blur Background (Keep Objects Sharp)")
        self.blur_checkbox.stateChanged.connect(lambda: setattr(self, 'apply_blur', self.blur_checkbox.isChecked()))
        
        self.trails_checkbox = QCheckBox("Show Object Movement Trails")
        self.trails_checkbox.stateChanged.connect(lambda: setattr(self, 'show_trails', self.trails_checkbox.isChecked()))
        
        effects_layout.addWidget(self.blur_checkbox)
        effects_layout.addWidget(self.trails_checkbox)
        
        # Blur intensity
        blur_intensity_layout = QHBoxLayout()
        blur_intensity_layout.addWidget(QLabel("Blur Intensity:"))
        self.blur_slider = QSlider(Qt.Orientation.Horizontal)
        self.blur_slider.setMinimum(1)
        self.blur_slider.setMaximum(51)
        self.blur_slider.setValue(21)
        self.blur_slider.setSingleStep(2)  # Odd numbers only for blur
        blur_intensity_layout.addWidget(self.blur_slider)
        effects_layout.addLayout(blur_intensity_layout)
        
        effects_group.setLayout(effects_layout)
        return effects_group

    def create_recording_controls(self):
        """Create recording control group"""
        recording_group = QGroupBox("🎬 Recording Controls")
        recording_layout = QVBoxLayout()
        
        self.snapshot_btn = QPushButton("📷 Save Snapshot (Current Frame)")
        self.snapshot_btn.clicked.connect(self.save_snapshot)
        
        self.record_btn = QPushButton("🔴 Start Video Recording")
        self.record_btn.clicked.connect(self.toggle_recording)
        
        self.batch_snapshot_btn = QPushButton("📸 Capture 10 Snapshots (1s interval)")
        self.batch_snapshot_btn.clicked.connect(self.batch_snapshot)
        
        recording_layout.addWidget(self.snapshot_btn)
        recording_layout.addWidget(self.record_btn)
        recording_layout.addWidget(self.batch_snapshot_btn)
        
        # Recording status
        self.recording_status = QLabel("⚫ Not Recording")
        self.recording_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.recording_status.setStyleSheet("background-color: #16213e; padding: 5px; border-radius: 3px;")
        recording_layout.addWidget(self.recording_status)
        
        recording_group.setLayout(recording_layout)
        return recording_group

    def create_timelapse_controls(self):
        """Create time-lapse control group"""
        timelapse_group = QGroupBox("⏱️ Time-lapse Features")
        timelapse_layout = QGridLayout()
        
        timelapse_layout.addWidget(QLabel("Interval (seconds):"), 0, 0)
        self.timelapse_interval_spin = QSpinBox()
        self.timelapse_interval_spin.setRange(1, 60)
        self.timelapse_interval_spin.setValue(2)
        self.timelapse_interval_spin.valueChanged.connect(lambda v: setattr(self, 'timelapse_interval', v))
        timelapse_layout.addWidget(self.timelapse_interval_spin, 0, 1)
        
        self.timelapse_btn = QPushButton("⏱️ Start Time-lapse")
        self.timelapse_btn.clicked.connect(self.toggle_timelapse)
        timelapse_layout.addWidget(self.timelapse_btn, 1, 0, 1, 2)
        
        self.timelapse_status = QLabel("⚫ Inactive | Frames: 0")
        self.timelapse_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.timelapse_status.setStyleSheet("background-color: #16213e; padding: 5px; border-radius: 3px;")
        timelapse_layout.addWidget(self.timelapse_status, 2, 0, 1, 2)
        
        self.save_timelapse_btn = QPushButton("💾 Save Time-lapse Video")
        self.save_timelapse_btn.clicked.connect(self.save_timelapse_video)
        self.save_timelapse_btn.setEnabled(False)
        timelapse_layout.addWidget(self.save_timelapse_btn, 3, 0, 1, 2)
        
        timelapse_group.setLayout(timelapse_layout)
        return timelapse_group

    def create_motion_controls(self):
        """Create motion detection control group"""
        motion_group = QGroupBox("🎭 Motion Detection")
        motion_layout = QGridLayout()
        
        self.motion_checkbox = QCheckBox("Enable Motion Detection")
        self.motion_checkbox.stateChanged.connect(self.toggle_motion_detection)
        motion_layout.addWidget(self.motion_checkbox, 0, 0, 1, 2)
        
        motion_layout.addWidget(QLabel("Sensitivity:"), 1, 0)
        self.motion_slider = QSlider(Qt.Orientation.Horizontal)
        self.motion_slider.setMinimum(1)
        self.motion_slider.setMaximum(100)
        self.motion_slider.setValue(25)
        self.motion_slider.valueChanged.connect(lambda v: setattr(self, 'motion_threshold', v))
        motion_layout.addWidget(self.motion_slider, 1, 1)
        
        self.motion_status = QLabel("⚫ No Motion Detected")
        self.motion_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.motion_status.setStyleSheet("background-color: #16213e; padding: 5px; border-radius: 3px;")
        motion_layout.addWidget(self.motion_status, 2, 0, 1, 2)
        
        motion_group.setLayout(motion_layout)
        return motion_group

    def create_alert_controls(self):
        """Create alert system control group"""
        alert_group = QGroupBox("🚨 Smart Alerts")
        alert_layout = QVBoxLayout()
        
        alert_layout.addWidget(QLabel("Alert when detecting:"))
        
        # Common objects for alerts
        alert_objects_layout = QGridLayout()
        self.alert_checkboxes = {}
        alert_items = ["person", "car", "dog", "cat", "bird", "cell phone"]
        
        for i, item in enumerate(alert_items):
            checkbox = QCheckBox(item.title())
            checkbox.stateChanged.connect(lambda state, obj=item: self.update_alert_objects(obj, state))
            alert_objects_layout.addWidget(checkbox, i // 2, i % 2)
            self.alert_checkboxes[item] = checkbox
        
        alert_layout.addLayout(alert_objects_layout)
        
        self.alert_sound_checkbox = QCheckBox("🔊 Enable Alert Sound")
        self.alert_sound_checkbox.stateChanged.connect(lambda: setattr(self, 'alert_sound_enabled', self.alert_sound_checkbox.isChecked()))
        alert_layout.addWidget(self.alert_sound_checkbox)
        
        self.alert_status = QLabel("✅ No Alerts")
        self.alert_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.alert_status.setStyleSheet("background-color: #16213e; padding: 5px; border-radius: 3px;")
        alert_layout.addWidget(self.alert_status)
        
        alert_group.setLayout(alert_layout)
        return alert_group

    def create_export_controls(self):
        """Create export control group"""
        export_group = QGroupBox("💾 Export & Analytics")
        export_layout = QVBoxLayout()
        
        self.export_stats_btn = QPushButton("📊 Export Session Statistics")
        self.export_stats_btn.clicked.connect(self.export_statistics)
        
        self.export_detections_btn = QPushButton("📋 Export Detection History (JSON)")
        self.export_detections_btn.clicked.connect(self.export_detections)
        
        self.reset_stats_btn = QPushButton("🔄 Reset All Statistics")
        self.reset_stats_btn.clicked.connect(self.reset_statistics)
        
        export_layout.addWidget(self.export_stats_btn)
        export_layout.addWidget(self.export_detections_btn)
        export_layout.addWidget(self.reset_stats_btn)
        
        export_group.setLayout(export_layout)
        return export_group

    def create_statistics_display(self):
        """Create statistics display group"""
        stats_group = QGroupBox("📊 Real-time Statistics")
        stats_layout = QGridLayout()
        
        self.fps_stat = QLabel("FPS: 0.0")
        self.fps_stat.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        
        self.total_objects_stat = QLabel("Total Objects: 0")
        self.total_objects_stat.setFont(QFont("Arial", 10))
        
        self.processing_time_stat = QLabel("Processing: 0ms")
        self.processing_time_stat.setFont(QFont("Arial", 10))
        
        self.session_time_stat = QLabel("Session: 0:00:00")
        self.session_time_stat.setFont(QFont("Arial", 10))
        
        self.detection_rate_stat = QLabel("Detection Rate: 0/s")
        self.detection_rate_stat.setFont(QFont("Arial", 10))
        
        stats_layout.addWidget(self.fps_stat, 0, 0)
        stats_layout.addWidget(self.total_objects_stat, 0, 1)
        stats_layout.addWidget(self.processing_time_stat, 1, 0)
        stats_layout.addWidget(self.session_time_stat, 1, 1)
        stats_layout.addWidget(self.detection_rate_stat, 2, 0, 1, 2)
        
        # Progress bar for system load (visual indicator)
        self.system_load_bar = QProgressBar()
        self.system_load_bar.setMaximum(100)
        self.system_load_bar.setValue(0)
        self.system_load_bar.setFormat("System Load: %p%")
        stats_layout.addWidget(self.system_load_bar, 3, 0, 1, 2)
        
        stats_group.setLayout(stats_layout)
        return stats_group

    def append_log(self, message):
        """Add message to activity log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def show_message_box(self, title, message):
        """Show message box dialog"""
        QMessageBox.information(self, title, message)

    def init_camera(self, camera_source):
        """Initialize camera with improved Continuity Camera support"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        self.camera_error_label.setText("🔄 Initializing camera...")
        
        try:
            if sys.platform == "darwin":  # macOS
                if camera_source == "continuity":
                    # Try to find Continuity Camera by scanning indices
                    self.append_log("🔍 Searching for Continuity Camera...")
                    camera_found = False
                    
                    for i in range(1, 10):  # Start from 1, as 0 is usually built-in
                        test_cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
                        if test_cap.isOpened():
                            ret, frame = test_cap.read()
                            if ret:
                                # Found a working camera
                                self.cap = test_cap
                                camera_found = True
                                self.append_log(f"✅ Found camera at index {i}")
                                break
                            else:
                                test_cap.release()
                        else:
                            test_cap.release()
                    
                    if not camera_found:
                        raise Exception("No Continuity Camera found. Please ensure iPhone is connected and Camera Continuity is enabled in System Preferences.")
                
                elif isinstance(camera_source, str):
                    self.cap = cv2.VideoCapture(camera_source, cv2.CAP_AVFOUNDATION)
                else:
                    self.cap = cv2.VideoCapture(camera_source, cv2.CAP_AVFOUNDATION)
            else:
                self.cap = cv2.VideoCapture(camera_source)
            
            if self.cap is None or not self.cap.isOpened():
                raise Exception("Failed to open camera")
            
            # Set resolution
            res_text = self.resolution_combo.currentText().split()[0]
            width, height = map(int, res_text.split('x'))
            self.width, self.height = set_camera_resolution(self.cap, width, height)
            
            # Test read
            ret, test_frame = self.cap.read()
            if not ret:
                raise Exception("Camera opened but cannot read frames")
            
            self.camera_error_label.setText("")
            self.signal_emitter.update_log.emit(f"✅ Camera initialized: {camera_source} ({self.width}x{self.height})")
            
        except Exception as e:
            if self.cap is not None:
                self.cap.release()
            self.cap = None
            error_message = f"❌ Camera error: {str(e)}"
            self.camera_error_label.setText(error_message)
            self.signal_emitter.update_log.emit(error_message)
            self.width, self.height = 640, 480

    def refresh_cameras(self):
        """Refresh available cameras"""
        self.append_log("🔄 Refreshing camera list...")
        self.camera_combo.clear()
        available_cameras = get_available_cameras()
        for name, index in available_cameras:
            self.camera_combo.addItem(name, index)
        self.append_log(f"✅ Found {len(available_cameras)} camera(s)")

    def capture_frames(self):
        """Capture frames in background thread"""
        while self.running:
            if self.cap is None:
                time.sleep(0.1)
                continue
            
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame.copy()
                self.detect_objects(frame)
            else:
                time.sleep(0.01)

    def detect_objects(self, frame):
        """Detect objects using YOLO"""
        global model
        start_time = time.time()
        
        try:
            # Use tracking if enabled
            if self.track_objects:
                results = model.track(
                    frame, 
                    device="mps", 
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    persist=True,
                    verbose=False
                )
            else:
                results = model(
                    frame, 
                    device="mps", 
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )
            
            processing_time = (time.time() - start_time) * 1000
            
            with self.lock:
                self.results = results
                self.processing_time = processing_time
                
                # Store detection history for analytics
                for result in results[0].boxes:
                    label = int(result.cls.cpu().numpy())
                    label_name = model.names[label]
                    self.detection_history[label_name].append(time.time())
                    self.object_history.append({
                        'time': time.time(),
                        'label': label_name,
                        'confidence': float(result.conf.cpu().numpy())
                    })
                
        except Exception as e:
            self.signal_emitter.update_log.emit(f"❌ Detection error: {str(e)}")

    def update_frame(self):
        """Update display frame"""
        with self.lock:
            frame = self.frame.copy() if self.frame is not None else None
            results = self.results
            processing_time = getattr(self, 'processing_time', 0)

        if frame is None or results is None:
            return

        # Motion detection
        if self.motion_detection:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            if self.last_frame_gray is not None:
                frame_delta = cv2.absdiff(self.last_frame_gray, gray)
                thresh = cv2.threshold(frame_delta, self.motion_threshold, 255, cv2.THRESH_BINARY)[1]
                motion_pixels = np.sum(thresh == 255)
                
                if motion_pixels > 1000:
                    self.motion_status.setText("🟢 Motion Detected!")
                    self.motion_status.setStyleSheet("background-color: #2ecc71; color: white; padding: 5px; border-radius: 3px; font-weight: bold;")
                else:
                    self.motion_status.setText("⚫ No Motion")
                    self.motion_status.setStyleSheet("background-color: #16213e; padding: 5px; border-radius: 3px;")
            
            self.last_frame_gray = gray

        # Apply background blur if enabled
        if self.apply_blur and len(results[0].boxes) > 0:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            for result in results[0].boxes:
                box = result.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(mask, (box[0], box[1]), (box[2], box[3]), 255, -1)
            
            blur_value = self.blur_slider.value()
            if blur_value % 2 == 0:  # Ensure odd number
                blur_value += 1
            blurred = cv2.GaussianBlur(frame, (blur_value, blur_value), 0)
            frame = np.where(mask[:,:,None] == 255, frame, blurred)

        # Object counting and visualization
        object_count = 0
        class_counts = {}
        detected_alert_objects = []
        
        for result in results[0].boxes:
            box = result.xyxy[0].cpu().numpy().astype(int)
            score = float(result.conf.cpu().numpy())
            label = int(result.cls.cpu().numpy())
            label_name = model.names[label]
            
            if score >= self.conf_threshold:
                object_count += 1
                class_counts[label_name] = class_counts.get(label_name, 0) + 1
                
                # Check for alerts
                if label_name in self.alert_objects:
                    detected_alert_objects.append(label_name)
                
                color = self.get_color_for_class(label)
                
                # Draw bounding box
                if self.show_boxes:
                    if self.fancy_boxes:
                        draw_fancy_box(frame, box, label_name, color, score)
                    else:
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                
                # Draw label and confidence
                if self.show_labels or self.show_confidence:
                    label_text = ""
                    if self.show_labels:
                        label_text += label_name
                    if self.show_confidence:
                        label_text += f" {score:.2f}" if self.show_labels else f"{score:.2f}"
                    
                    # Draw label background
                    (label_width, label_height), _ = cv2.getTextSize(
                        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(
                        frame, 
                        (box[0], box[1] - label_height - 10),
                        (box[0] + label_width + 10, box[1]),
                        color, -1
                    )
                    cv2.putText(
                        frame, label_text,
                        (box[0] + 5, box[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                    )

        # Handle alerts
        if detected_alert_objects:
            alert_text = f"🚨 ALERT: {', '.join(set(detected_alert_objects))}"
            self.alert_status.setText(alert_text)
            self.alert_status.setStyleSheet("background-color: #e74c3c; color: white; padding: 5px; border-radius: 3px; font-weight: bold;")
        else:
            self.alert_status.setText("✅ No Alerts")
            self.alert_status.setStyleSheet("background-color: #16213e; padding: 5px; border-radius: 3px;")

        # FPS calculation with moving average
        current_time = time.time()
        dt = current_time - self.last_time
        if dt > 0:
            current_fps = 1 / dt
            self.fps_history.append(current_fps)
            self.fps = sum(self.fps_history) / len(self.fps_history)
        self.last_time = current_time
        
        # Overlay FPS on video
        if self.show_fps_overlay:
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Objects: {object_count}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Update quick stats
        self.quick_fps.setText(f"⚡ FPS: {self.fps:.1f}")
        self.quick_objects.setText(f"🎯 Objects: {object_count}")
        
        # Update statistics panel
        self.fps_stat.setText(f"FPS: {self.fps:.1f}")
        self.total_objects_stat.setText(f"Total Objects: {object_count}")
        self.processing_time_stat.setText(f"Processing: {processing_time:.1f}ms")
        
        # Calculate system load (based on processing time vs frame time)
        target_frame_time = 1000 / self.fps_target_spin.value()
        load_percent = min(100, int((processing_time / target_frame_time) * 100))
        self.system_load_bar.setValue(load_percent)
        
        if class_counts:
            counts_text = " | ".join([f"{k}:{v}" for k, v in sorted(class_counts.items())])
            self.total_objects_stat.setText(f"Total: {object_count} | {counts_text}")

        # Time-lapse capture
        if self.timelapse_active:
            if current_time - self.timelapse_last_capture >= self.timelapse_interval:
                self.timelapse_frames.append(frame.copy())
                self.timelapse_counter += 1
                self.timelapse_last_capture = current_time
                self.timelapse_status.setText(f"🟢 Recording | Frames: {self.timelapse_counter}")
                self.append_log(f"⏱️ Time-lapse frame {self.timelapse_counter} captured")

        # Record video if recording
        if self.recording and self.video_writer is not None:
            self.video_writer.write(frame)

        # Display frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    def update_analytics(self):
        """Update analytics every 5 seconds"""
        session_duration = time.time() - self.session_start_time
        hours = int(session_duration // 3600)
        minutes = int((session_duration % 3600) // 60)
        seconds = int(session_duration % 60)
        self.session_time_stat.setText(f"Session: {hours}:{minutes:02d}:{seconds:02d}")
        
        # Calculate detection rate
        recent_detections = sum(1 for obj in self.object_history 
                               if time.time() - obj['time'] < 5)
        detection_rate = recent_detections / 5.0
        self.detection_rate_stat.setText(f"Detection Rate: {detection_rate:.1f}/s")

    def get_color_for_class(self, class_id):
        """Generate consistent color for each class"""
        np.random.seed(class_id * 42)
        return tuple(np.random.randint(100, 255, 3).tolist())

    def save_snapshot(self):
        """Save current frame as snapshot"""
        with self.lock:
            if self.frame is not None:
                os.makedirs("snapshots", exist_ok=True)
                filename = f"snapshots/snapshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(filename, self.frame)
                self.signal_emitter.update_log.emit(f"📷 Snapshot saved: {filename}")
                self.signal_emitter.show_message.emit("Success", f"Snapshot saved to:\n{filename}")

    def batch_snapshot(self):
        """Capture multiple snapshots with interval"""
        self.append_log("📸 Starting batch snapshot capture...")
        self.batch_snapshot_btn.setEnabled(False)
        
        def capture_batch():
            for i in range(10):
                time.sleep(1)
                with self.lock:
                    if self.frame is not None:
                        os.makedirs("snapshots", exist_ok=True)
                        filename = f"snapshots/batch_{time.strftime('%Y%m%d_%H%M%S')}_{i+1}.png"
                        cv2.imwrite(filename, self.frame)
                        self.signal_emitter.update_log.emit(f"📸 Batch snapshot {i+1}/10 saved")
            
            self.signal_emitter.update_log.emit("✅ Batch snapshot complete!")
            self.batch_snapshot_btn.setEnabled(True)
        
        thread = threading.Thread(target=capture_batch, daemon=True)
        thread.start()

    def toggle_recording(self):
        """Toggle video recording"""
        self.recording = not self.recording
        if self.recording:
            os.makedirs("videos", exist_ok=True)
            filename = f"videos/recording_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (self.width, self.height))
            self.record_btn.setText("⏹️ Stop Recording")
            self.record_btn.setStyleSheet("background-color: #e74c3c; border: 2px solid #c0392b;")
            self.recording_status.setText("🔴 Recording...")
            self.recording_status.setStyleSheet("background-color: #e74c3c; color: white; padding: 5px; border-radius: 3px; font-weight: bold;")
            self.signal_emitter.update_log.emit(f"🎬 Recording started: {filename}")
        else:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self.record_btn.setText("🔴 Start Video Recording")
            self.record_btn.setStyleSheet("")
            self.recording_status.setText("⚫ Not Recording")
            self.recording_status.setStyleSheet("background-color: #16213e; padding: 5px; border-radius: 3px;")
            self.signal_emitter.update_log.emit("✅ Recording stopped and saved")

    def toggle_timelapse(self):
        """Toggle time-lapse recording"""
        self.timelapse_active = not self.timelapse_active
        
        if self.timelapse_active:
            self.timelapse_frames = []
            self.timelapse_counter = 0
            self.timelapse_last_capture = time.time()
            self.timelapse_btn.setText("⏹️ Stop Time-lapse")
            self.timelapse_btn.setStyleSheet("background-color: #e74c3c; border: 2px solid #c0392b;")
            self.timelapse_status.setText("🟢 Recording | Frames: 0")
            self.timelapse_status.setStyleSheet("background-color: #2ecc71; color: white; padding: 5px; border-radius: 3px; font-weight: bold;")
            self.save_timelapse_btn.setEnabled(False)
            self.signal_emitter.update_log.emit(f"⏱️ Time-lapse started (interval: {self.timelapse_interval}s)")
        else:
            self.timelapse_btn.setText("⏱️ Start Time-lapse")
            self.timelapse_btn.setStyleSheet("")
            self.timelapse_status.setText(f"⚫ Stopped | Frames: {self.timelapse_counter}")
            self.timelapse_status.setStyleSheet("background-color: #16213e; padding: 5px; border-radius: 3px;")
            self.save_timelapse_btn.setEnabled(True if self.timelapse_counter > 0 else False)
            self.signal_emitter.update_log.emit(f"✅ Time-lapse stopped ({self.timelapse_counter} frames captured)")

    def save_timelapse_video(self):
        """Save captured time-lapse frames as video"""
        if not self.timelapse_frames:
            self.signal_emitter.show_message.emit("Error", "No time-lapse frames to save!")
            return
        
        self.append_log("💾 Saving time-lapse video...")
        self.save_timelapse_btn.setEnabled(False)
        
        def save_video():
            try:
                os.makedirs("videos", exist_ok=True)
                filename = f"videos/timelapse_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                
                # Use first frame dimensions
                height, width = self.timelapse_frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(filename, fourcc, 10.0, (width, height))
                
                for frame in self.timelapse_frames:
                    writer.write(frame)
                
                writer.release()
                self.signal_emitter.update_log.emit(f"✅ Time-lapse video saved: {filename}")
                self.signal_emitter.show_message.emit("Success", f"Time-lapse video saved to:\n{filename}\n\nFrames: {len(self.timelapse_frames)}")
                
            except Exception as e:
                self.signal_emitter.update_log.emit(f"❌ Error saving time-lapse: {str(e)}")
                self.signal_emitter.show_message.emit("Error", f"Failed to save time-lapse:\n{str(e)}")
            finally:
                self.save_timelapse_btn.setEnabled(True)
        
        thread = threading.Thread(target=save_video, daemon=True)
        thread.start()

    def toggle_motion_detection(self, state):
        """Toggle motion detection"""
        self.motion_detection = state == Qt.CheckState.Checked.value
        if self.motion_detection:
            self.append_log("🎭 Motion detection enabled")
        else:
            self.append_log("⚫ Motion detection disabled")
            self.motion_status.setText("⚫ Disabled")
            self.motion_status.setStyleSheet("background-color: #16213e; padding: 5px; border-radius: 3px;")

    def update_alert_objects(self, obj, state):
        """Update alert object list"""
        if state == Qt.CheckState.Checked.value:
            self.alert_objects.add(obj)
            self.append_log(f"🚨 Alert enabled for: {obj}")
        else:
            self.alert_objects.discard(obj)
            self.append_log(f"⚫ Alert disabled for: {obj}")

    def export_log(self):
        """Export activity log to file"""
        try:
            os.makedirs("exports", exist_ok=True)
            filename = f"exports/log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w') as f:
                f.write(self.log_text.toPlainText())
            self.append_log(f"📄 Log exported: {filename}")
            self.signal_emitter.show_message.emit("Success", f"Log exported to:\n{filename}")
        except Exception as e:
            self.append_log(f"❌ Export error: {str(e)}")

    def export_statistics(self):
        """Export session statistics"""
        try:
            os.makedirs("exports", exist_ok=True)
            filename = f"exports/stats_{time.strftime('%Y%m%d_%H%M%S')}.json"

            # Compile statistics
            stats = {
                "session_duration": time.time() - self.session_start_time,
                "average_fps": self.fps,
                "total_detections": len(self.object_history),
                "detection_by_class": {},
                "confidence_stats": {
                    "min": min([obj['confidence'] for obj in self.object_history]) if self.object_history else 0,
                    "max": max([obj['confidence'] for obj in self.object_history]) if self.object_history else 0,
                    "avg": sum([obj['confidence'] for obj in self.object_history]) / len(self.object_history) if self.object_history else 0
                },
                "model_used": current_model_name,
                "resolution": f"{self.width}x{self.height}",
                "timestamp": datetime.now().isoformat()
            }

            # Count detections by class
            for obj in self.object_history:
                label = obj['label']
                stats['detection_by_class'][label] = stats['detection_by_class'].get(label, 0) + 1

            with open(filename, 'w') as f:
                json.dump(stats, f, indent=4)

            self.append_log(f"📊 Statistics exported: {filename}")
            self.signal_emitter.show_message.emit("Success", f"Statistics exported to:\n{filename}")
        except Exception as e:
            self.append_log(f"❌ Export error: {str(e)}")

    def export_detections(self):
        """Export detection history as JSON"""
        try:
            os.makedirs("exports", exist_ok=True)
            filename = f"exports/detections_{time.strftime('%Y%m%d_%H%M%S')}.json"

            # Prepare detection data
            detections = []
            for obj in self.object_history:
                detections.append({
                    "timestamp": datetime.fromtimestamp(obj['time']).isoformat(),
                    "label": obj['label'],
                    "confidence": round(obj['confidence'], 3)
                })

            with open(filename, 'w') as f:
                json.dump(detections, f, indent=2)

            self.append_log(f"📋 Detections exported: {filename}")
            self.signal_emitter.show_message.emit(
                "Success",
                f"Detection history exported to:\n{filename}\n\nTotal detections: {len(detections)}"
            )
        except Exception as e:
            self.append_log(f"❌ Export error: {str(e)}")


    def reset_statistics(self):
        """Reset all statistics"""
        reply = QMessageBox.question(
            self,
            'Reset Statistics',
            'Are you sure you want to reset all statistics?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.object_history.clear()
            self.detection_history.clear()
            self.session_start_time = time.time()
            self.fps_history.clear()
            self.append_log("🔄 Statistics reset successfully")

    def update_confidence(self, value):
        """Update confidence threshold"""
        self.conf_threshold = value / 100
        self.conf_label.setText(f"{self.conf_threshold:.2f}")

    def update_iou(self, value):
        """Update IOU threshold"""
        self.iou_threshold = value / 100
        self.iou_label.setText(f"{self.iou_threshold:.2f}")

    def switch_model(self, model_name):
        """Switch YOLO model"""
        global model, current_model_name
        try:
            self.append_log(f"🔄 Loading model: {model_name}...")
            model = YOLO(YOLO_MODELS[model_name])
            current_model_name = model_name
            self.quick_model.setText(f"🤖 Model: {model_name}")
            self.signal_emitter.update_log.emit(f"✅ Model loaded: {model_name}")
        except Exception as e:
            self.signal_emitter.update_log.emit(f"❌ Model switch error: {str(e)}")

    def switch_camera(self, index):
        """Switch camera source"""
        camera_source = self.camera_combo.itemData(index)
        if self.recording:
            self.toggle_recording()
        if self.timelapse_active:
            self.toggle_timelapse()
        self.append_log(f"🔄 Switching to camera: {camera_source}")
        self.init_camera(camera_source)

    def change_resolution(self, resolution_text):
        """Change camera resolution"""
        if self.cap is not None:
            res_text = resolution_text.split()[0]
            width, height = map(int, res_text.split('x'))
            self.width, self.height = set_camera_resolution(self.cap, width, height)
            self.signal_emitter.update_log.emit(f"📐 Resolution changed to: {res_text}")

    def closeEvent(self, event):
        """Handle application close"""
        self.running = False

        # Stop recording if active
        if self.recording and self.video_writer is not None:
            self.video_writer.release()

        # Release camera
        if self.cap is not None:
            self.cap.release()

        self.signal_emitter.update_log.emit("👋 Application closed")
        event.accept()


# ----------------------------
# Main Application Entry
# ----------------------------
def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application metadata
    app.setApplicationName("YOLOv8 Pro Max")
    app.setOrganizationName("AI Vision Studio")
    app.setApplicationVersion("2.0.0")
    
    # Create and show main window
    widget = VideoCaptureWidget()
    widget.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()