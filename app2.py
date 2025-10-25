import sys
import cv2
import threading
import time
from ultralytics import YOLO
from PyQt6.QtCore import QTimer, Qt, QPoint
from PyQt6.QtWidgets import (
    QApplication, QLabel, QVBoxLayout, QWidget, QPushButton, QHBoxLayout,
    QComboBox, QSlider, QGridLayout, QCheckBox, QGroupBox, QScrollArea,
    QFrame, QSizePolicy
)
from PyQt6.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QColor
from collections import deque
import os
import numpy as np
from datetime import datetime
import json

# ----------------------------
# YOLO Models Configuration
# ----------------------------
YOLO_MODELS = {
    "YOLOv8n": "yolov8n.pt",
    "YOLOv8m": "yolov8m.pt",
    "YOLOv8l": "yolov8l.pt"
}
current_model_name = "YOLOv8m"
model = None

# ----------------------------
# Camera Resolution Function
# ----------------------------
def set_camera_resolution(cap, width=1920, height=1080):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution set to: {actual_width}x{actual_height}")
    return actual_width, actual_height

# ----------------------------
# Analytics Data Manager
# ----------------------------
class AnalyticsManager:
    def __init__(self, max_history=100):
        self.max_history = max_history
        self.detection_history = deque(maxlen=max_history)
        self.class_totals = {}
        self.timestamps = deque(maxlen=max_history)
        self.start_time = time.time()
        
    def add_detection(self, class_counts):
        """Add detection data point"""
        total = sum(class_counts.values())
        self.detection_history.append(total)
        self.timestamps.append(time.time() - self.start_time)
        
        # Update class totals
        for cls, count in class_counts.items():
            self.class_totals[cls] = self.class_totals.get(cls, 0) + count
    
    def get_history(self):
        """Get detection history"""
        return list(self.detection_history)
    
    def get_class_distribution(self):
        """Get class distribution for pie chart"""
        return self.class_totals.copy()
    
    def export_report(self, filename):
        """Export analytics report"""
        report = {
            "session_duration": time.time() - self.start_time,
            "total_detections": sum(self.detection_history),
            "class_distribution": self.class_totals,
            "peak_detections": max(self.detection_history) if self.detection_history else 0,
            "average_detections": sum(self.detection_history) / len(self.detection_history) if self.detection_history else 0
        }
        
        os.makedirs("reports", exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(report, f, indent=4)
        print(f"Report exported: {filename}")

# ----------------------------
# Zone Manager for ROI
# ----------------------------
class ZoneManager:
    def __init__(self):
        self.zones = []
        self.drawing = False
        self.current_zone = []
        self.zone_counts = {}
        
    def start_zone(self, point):
        """Start drawing a new zone"""
        self.drawing = True
        self.current_zone = [point]
    
    def add_point(self, point):
        """Add point to current zone"""
        if self.drawing:
            self.current_zone.append(point)
    
    def finish_zone(self):
        """Finish drawing current zone"""
        if len(self.current_zone) >= 3:
            self.zones.append(self.current_zone.copy())
            zone_id = len(self.zones) - 1
            self.zone_counts[zone_id] = 0
        self.drawing = False
        self.current_zone = []
    
    def clear_zones(self):
        """Clear all zones"""
        self.zones = []
        self.zone_counts = {}
        self.current_zone = []
        self.drawing = False
    
    def point_in_zone(self, point, zone_points):
        """Check if point is inside zone using ray casting"""
        x, y = point
        n = len(zone_points)
        inside = False
        
        p1x, p1y = zone_points[0]
        for i in range(1, n + 1):
            p2x, p2y = zone_points[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    def count_objects_in_zones(self, detections):
        """Count objects in each zone"""
        self.zone_counts = {i: 0 for i in range(len(self.zones))}
        
        for detection in detections:
            box = detection['box']
            center_x = (box[0] + box[2]) // 2
            center_y = (box[1] + box[3]) // 2
            
            for zone_id, zone_points in enumerate(self.zones):
                if self.point_in_zone((center_x, center_y), zone_points):
                    self.zone_counts[zone_id] += 1
        
        return self.zone_counts

# ----------------------------
# Alert Manager
# ----------------------------
class AlertManager:
    def __init__(self):
        self.alert_classes = set()
        self.alert_threshold = 1
        self.alerts_triggered = []
        
    def add_alert_class(self, class_name):
        """Add class to alert list"""
        self.alert_classes.add(class_name)
    
    def remove_alert_class(self, class_name):
        """Remove class from alert list"""
        self.alert_classes.discard(class_name)
    
    def check_alerts(self, class_counts):
        """Check if any alerts should be triggered"""
        triggered = []
        for cls in self.alert_classes:
            count = class_counts.get(cls, 0)
            if count >= self.alert_threshold:
                triggered.append((cls, count))
        
        self.alerts_triggered = triggered
        return triggered

# ----------------------------
# Main GUI Class
# ----------------------------
class VideoCaptureWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8 Pro - Advanced Object Detection")
        self.setGeometry(50, 50, 1600, 900)
        
        # Apply dark theme stylesheet
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
                font-family: 'Segoe UI', Arial;
            }
            QGroupBox {
                border: 2px solid #3d3d3d;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: bold;
                color: #00d4ff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #0d7377;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                color: white;
            }
            QPushButton:hover {
                background-color: #14a085;
            }
            QPushButton:pressed {
                background-color: #0a5f63;
            }
            QPushButton:disabled {
                background-color: #3d3d3d;
                color: #666666;
            }
            QComboBox, QSlider {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 5px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #00d4ff;
            }
            QLabel {
                color: #ffffff;
            }
            QCheckBox {
                color: #ffffff;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #3d3d3d;
                border-radius: 4px;
                background-color: #2d2d2d;
            }
            QCheckBox::indicator:checked {
                background-color: #00d4ff;
                border-color: #00d4ff;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #3d3d3d;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #00d4ff;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QScrollArea {
                border: none;
                background-color: #1e1e1e;
            }
        """)

        # Initialize managers
        self.analytics = AnalyticsManager()
        self.zone_manager = ZoneManager()
        self.alert_manager = AlertManager()

        # Initialize camera
        self.camera_index = 0
        self.cap = None
        self.width = 0
        self.height = 0
        
        # Initialize YOLO model
        global model
        model = YOLO(YOLO_MODELS[current_model_name])

        # Feature flags
        self.show_trails = False
        self.show_heatmap = False
        self.show_zones = True
        self.show_analytics = True
        self.filter_enabled = False
        self.filtered_classes = set()
        
        # Object tracking
        self.object_trails = {}  # {object_id: deque of positions}
        self.trail_length = 30
        
        # Heatmap
        self.heatmap = None
        self.heatmap_alpha = 0.4

        # Setup UI
        self.setup_ui()
        
        # Threading & FPS
        self.lock = threading.Lock()
        self.frame = None
        self.results = None
        self.last_time = time.time()
        self.fps = 0
        self.conf_threshold = 0.25
        self.running = True
        self.recording = False
        self.video_writer = None

        # Initialize camera
        self.init_camera(0)

        # Start capture thread
        self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        self.capture_thread.start()

        # Timer for GUI update
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # ~30 FPS

    def setup_ui(self):
        """Setup the user interface"""
        # Main layout
        main_layout = QHBoxLayout()
        
        # Left panel - Video display
        left_panel = QVBoxLayout()
        
        self.image_label = ClickableLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("border: 2px solid #3d3d3d; border-radius: 8px; background-color: #000000;")
        self.image_label.mouse_pressed.connect(self.on_image_click)
        
        self.camera_error_label = QLabel("", self)
        self.camera_error_label.setFont(QFont("Arial", 12))
        self.camera_error_label.setStyleSheet("color: #ff4444;")
        self.camera_error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        left_panel.addWidget(self.image_label)
        left_panel.addWidget(self.camera_error_label)
        
        # Stats bar
        stats_layout = QHBoxLayout()
        self.fps_label = QLabel("FPS: 0", self)
        self.fps_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.fps_label.setStyleSheet("color: #00ff88;")
        
        self.count_label = QLabel("Objects: 0", self)
        self.count_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.count_label.setStyleSheet("color: #ffaa00;")
        
        stats_layout.addWidget(self.fps_label)
        stats_layout.addWidget(self.count_label)
        stats_layout.addStretch()
        
        left_panel.addLayout(stats_layout)
        
        # Right panel - Controls (scrollable)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setMaximumWidth(400)
        right_scroll.setMinimumWidth(350)
        
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        
        # Camera & Model Controls
        control_group = QGroupBox("🎥 Camera & Model")
        control_layout = QGridLayout()
        
        control_layout.addWidget(QLabel("Camera:"), 0, 0)
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("MacBook Camera", 0)
        self.camera_combo.addItem("iPhone Camera", "iPhone")
        self.camera_combo.currentIndexChanged.connect(self.switch_camera)
        control_layout.addWidget(self.camera_combo, 0, 1)
        
        control_layout.addWidget(QLabel("YOLO Model:"), 1, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(YOLO_MODELS.keys())
        self.model_combo.setCurrentText(current_model_name)
        self.model_combo.currentTextChanged.connect(self.switch_model)
        control_layout.addWidget(self.model_combo, 1, 1)
        
        control_layout.addWidget(QLabel("Confidence:"), 2, 0)
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setMinimum(1)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(25)
        self.conf_slider.valueChanged.connect(self.update_confidence)
        control_layout.addWidget(self.conf_slider, 2, 1)
        
        self.conf_label = QLabel("0.25")
        self.conf_label.setStyleSheet("color: #00d4ff; font-weight: bold;")
        control_layout.addWidget(self.conf_label, 2, 2)
        
        control_group.setLayout(control_layout)
        right_layout.addWidget(control_group)
        
        # Recording Controls
        record_group = QGroupBox("⏺️ Recording")
        record_layout = QVBoxLayout()
        
        self.snapshot_btn = QPushButton("📷 Save Snapshot")
        self.snapshot_btn.clicked.connect(self.save_snapshot)
        
        self.record_btn = QPushButton("🔴 Start Recording")
        self.record_btn.clicked.connect(self.toggle_recording)
        
        record_layout.addWidget(self.snapshot_btn)
        record_layout.addWidget(self.record_btn)
        record_group.setLayout(record_layout)
        right_layout.addWidget(record_group)
        
        # Visualization Options
        viz_group = QGroupBox("👁️ Visualization")
        viz_layout = QVBoxLayout()
        
        self.trails_check = QCheckBox("Show Object Trails")
        self.trails_check.stateChanged.connect(lambda: self.toggle_feature('trails'))
        
        self.heatmap_check = QCheckBox("Show Heatmap")
        self.heatmap_check.stateChanged.connect(lambda: self.toggle_feature('heatmap'))
        
        self.zones_check = QCheckBox("Show Zones")
        self.zones_check.setChecked(True)
        self.zones_check.stateChanged.connect(lambda: self.toggle_feature('zones'))
        
        viz_layout.addWidget(self.trails_check)
        viz_layout.addWidget(self.heatmap_check)
        viz_layout.addWidget(self.zones_check)
        viz_group.setLayout(viz_layout)
        right_layout.addWidget(viz_group)
        
        # Zone Controls
        zone_group = QGroupBox("📍 Zone Management")
        zone_layout = QVBoxLayout()
        
        self.zone_info_label = QLabel("Click image to draw zones\n(3+ points, double-click to finish)")
        self.zone_info_label.setWordWrap(True)
        self.zone_info_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        
        self.zone_count_label = QLabel("Zones: 0 | Objects in zones: 0")
        self.zone_count_label.setStyleSheet("color: #00ff88;")
        
        self.clear_zones_btn = QPushButton("🗑️ Clear All Zones")
        self.clear_zones_btn.clicked.connect(self.clear_zones)
        
        zone_layout.addWidget(self.zone_info_label)
        zone_layout.addWidget(self.zone_count_label)
        zone_layout.addWidget(self.clear_zones_btn)
        zone_group.setLayout(zone_layout)
        right_layout.addWidget(zone_group)
        
        # Alert System
        alert_group = QGroupBox("🔔 Alerts")
        alert_layout = QVBoxLayout()
        
        alert_layout.addWidget(QLabel("Alert on detection:"))
        self.alert_class_combo = QComboBox()
        self.alert_class_combo.addItem("Select class...")
        self.alert_class_combo.currentTextChanged.connect(self.add_alert)
        
        self.active_alerts_label = QLabel("Active alerts: None")
        self.active_alerts_label.setWordWrap(True)
        self.active_alerts_label.setStyleSheet("color: #ffaa00; font-size: 11px;")
        
        self.alert_status_label = QLabel("")
        self.alert_status_label.setStyleSheet("color: #ff4444; font-weight: bold; font-size: 13px;")
        
        alert_layout.addWidget(self.alert_class_combo)
        alert_layout.addWidget(self.active_alerts_label)
        alert_layout.addWidget(self.alert_status_label)
        alert_group.setLayout(alert_layout)
        right_layout.addWidget(alert_group)
        
        # Analytics
        analytics_group = QGroupBox("📊 Analytics")
        analytics_layout = QVBoxLayout()
        
        self.total_detections_label = QLabel("Total Detections: 0")
        self.avg_detections_label = QLabel("Avg per Frame: 0.0")
        self.peak_detections_label = QLabel("Peak: 0")
        
        self.export_report_btn = QPushButton("📄 Export Report")
        self.export_report_btn.clicked.connect(self.export_report)
        
        analytics_layout.addWidget(self.total_detections_label)
        analytics_layout.addWidget(self.avg_detections_label)
        analytics_layout.addWidget(self.peak_detections_label)
        analytics_layout.addWidget(self.export_report_btn)
        analytics_group.setLayout(analytics_layout)
        right_layout.addWidget(analytics_group)
        
        # Class Filter
        filter_group = QGroupBox("🔍 Class Filter")
        filter_layout = QVBoxLayout()
        
        self.filter_check = QCheckBox("Enable Filtering")
        self.filter_check.stateChanged.connect(self.toggle_filter)
        
        self.filter_info = QLabel("No filters active")
        self.filter_info.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        self.filter_info.setWordWrap(True)
        
        filter_layout.addWidget(self.filter_check)
        filter_layout.addWidget(self.filter_info)
        filter_group.setLayout(filter_layout)
        right_layout.addWidget(filter_group)
        
        right_layout.addStretch()
        right_widget.setLayout(right_layout)
        right_scroll.setWidget(right_widget)
        
        # Add panels to main layout
        main_layout.addLayout(left_panel, 7)
        main_layout.addWidget(right_scroll, 3)
        
        self.setLayout(main_layout)

    def init_camera(self, camera_source):
        """Initialize camera"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.camera_error_label.setText("")
        
        try:
            if sys.platform == "darwin":
                if isinstance(camera_source, str):
                    self.cap = cv2.VideoCapture(camera_source, cv2.CAP_AVFOUNDATION)
                else:
                    self.cap = cv2.VideoCapture(camera_source, cv2.CAP_AVFOUNDATION)
            else:
                self.cap = cv2.VideoCapture(camera_source)
            
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = None
                error_message = "Cannot open camera. Check permissions."
                print(error_message)
                self.camera_error_label.setText(error_message)
                self.width, self.height = 640, 480
                return
            
            self.width, self.height = set_camera_resolution(self.cap)
            self.heatmap = np.zeros((self.height, self.width), dtype=np.float32)
            self.camera_error_label.setText("")
        except Exception as e:
            if self.cap is not None:
                self.cap.release()
            self.cap = None
            error_message = f"Camera error: {e}"
            print(error_message)
            self.camera_error_label.setText(error_message)
            self.width, self.height = 640, 480

    def capture_frames(self):
        """Capture frames in separate thread"""
        while self.running:
            if self.cap is None:
                time.sleep(0.01)
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
        results = model(frame, device="mps", conf=self.conf_threshold)
        with self.lock:
            self.results = results

    def update_frame(self):
        """Update GUI with latest frame and detections"""
        with self.lock:
            frame = self.frame.copy() if self.frame is not None else None
            results = self.results

        if frame is None or results is None:
            return

        # Parse detections
        detections = []
        object_count = 0
        class_counts = {}
        
        for result in results[0].boxes:
            box = result.xyxy[0].cpu().numpy().astype(int)
            score = float(result.conf.cpu().numpy())
            label = int(result.cls.cpu().numpy())
            label_name = model.names[label]
            
            if score >= self.conf_threshold:
                # Apply filter if enabled
                if self.filter_enabled and label_name not in self.filtered_classes:
                    continue
                
                detection = {
                    'box': box,
                    'score': score,
                    'label': label_name,
                    'center': ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
                }
                detections.append(detection)
                
                object_count += 1
                class_counts[label_name] = class_counts.get(label_name, 0) + 1
                
                # Update heatmap
                if self.show_heatmap:
                    cv2.rectangle(self.heatmap, (box[0], box[1]), (box[2], box[3]), 1, -1)
                
                # Draw detection
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"{label_name}:{score:.2f}", (box[0], max(box[1]-10, 0)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Update analytics
        self.analytics.add_detection(class_counts)
        
        # Update alert combo with detected classes
        current_items = [self.alert_class_combo.itemText(i) for i in range(self.alert_class_combo.count())]
        for cls in class_counts.keys():
            if cls not in current_items:
                self.alert_class_combo.addItem(cls)
        
        # Check alerts
        alerts = self.alert_manager.check_alerts(class_counts)
        if alerts:
            alert_text = " | ".join([f"⚠️ {cls}: {count}" for cls, count in alerts])
            self.alert_status_label.setText(alert_text)
        else:
            self.alert_status_label.setText("")
        
        # Zone counting
        if self.show_zones and self.zone_manager.zones:
            zone_counts = self.zone_manager.count_objects_in_zones(detections)
            total_in_zones = sum(zone_counts.values())
            self.zone_count_label.setText(f"Zones: {len(self.zone_manager.zones)} | Objects in zones: {total_in_zones}")
        else:
            self.zone_count_label.setText(f"Zones: {len(self.zone_manager.zones)} | Objects in zones: 0")
        
        # Draw heatmap overlay
        if self.show_heatmap:
            heatmap_colored = cv2.applyColorMap((self.heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
            frame = cv2.addWeighted(frame, 1 - self.heatmap_alpha, heatmap_colored, self.heatmap_alpha, 0)
            self.heatmap *= 0.95  # Decay
        
        # Draw zones
        if self.show_zones:
            for i, zone in enumerate(self.zone_manager.zones):
                pts = np.array(zone, np.int32)
                cv2.polylines(frame, [pts], True, (255, 0, 255), 2)
                if zone:
                    cv2.putText(frame, f"Zone {i+1}", (zone[0][0], zone[0][1]-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            # Draw current zone being drawn
            if self.zone_manager.drawing and len(self.zone_manager.current_zone) > 0:
                pts = np.array(self.zone_manager.current_zone, np.int32)
                cv2.polylines(frame, [pts], False, (0, 255, 255), 2)
                for pt in self.zone_manager.current_zone:
                    cv2.circle(frame, pt, 5, (0, 255, 255), -1)
        
        # Update analytics labels
        history = self.analytics.get_history()
        if history:
            self.total_detections_label.setText(f"Total Detections: {sum(history)}")
            self.avg_detections_label.setText(f"Avg per Frame: {sum(history)/len(history):.1f}")
            self.peak_detections_label.setText(f"Peak: {max(history)}")
        
        # FPS calculation
        current_time = time.time()
        dt = current_time - self.last_time
        if dt > 0:
            self.fps = 1 / dt
        self.last_time = current_time
        self.fps_label.setText(f"FPS: {self.fps:.1f}")
        
        # Count label
        if class_counts:
            counts_text = ", ".join([f"{k}:{v}" for k,v in sorted(class_counts.items())])
            self.count_label.setText(f"Objects: {object_count} | {counts_text}")
        else:
            self.count_label.setText(f"Objects: {object_count}")
        
        # Recording
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

    def on_image_click(self, x, y):
        """Handle mouse click on image for zone drawing"""
        if not self.show_zones:
            return
        
        # Convert click coordinates to frame coordinates
        pixmap = self.image_label.pixmap()
        if pixmap is None:
            return
        
        # Calculate scaling
        label_w = self.image_label.width()
        label_h = self.image_label.height()
        pixmap_w = pixmap.width()
        pixmap_h = pixmap.height()
        
        # Calculate actual image position in label (centered with aspect ratio)
        scale = min(label_w / pixmap_w, label_h / pixmap_h)
        scaled_w = int(pixmap_w * scale)
        scaled_h = int(pixmap_h * scale)
        offset_x = (label_w - scaled_w) // 2
        offset_y = (label_h - scaled_h) // 2
        
        # Check if click is within image bounds
        if x < offset_x or x > offset_x + scaled_w or y < offset_y or y > offset_y + scaled_h:
            return
        
        # Convert to frame coordinates
        frame_x = int((x - offset_x) / scale)
        frame_y = int((y - offset_y) / scale)
        
        # Clamp to frame dimensions
        frame_x = max(0, min(frame_x, self.width - 1))
        frame_y = max(0, min(frame_y, self.height - 1))
        
        # Add point to zone
        if not self.zone_manager.drawing:
            self.zone_manager.start_zone((frame_x, frame_y))
        else:
            self.zone_manager.add_point((frame_x, frame_y))

    def on_image_double_click(self):
        """Finish zone drawing on double click"""
        if self.zone_manager.drawing:
            self.zone_manager.finish_zone()
            print(f"Zone created with {len(self.zone_manager.zones[-1])} points")

    def save_snapshot(self):
        """Save current frame as snapshot"""
        with self.lock:
            if self.frame is not None:
                os.makedirs("snapshots", exist_ok=True)
                filename = f"snapshots/snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(filename, self.frame)
                print(f"Snapshot saved: {filename}")

    def toggle_recording(self):
        """Toggle video recording"""
        self.recording = not self.recording
        if self.recording:
            os.makedirs("videos", exist_ok=True)
            filename = f"videos/recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (self.width, self.height))
            self.record_btn.setText("⏹️ Stop Recording")
            self.record_btn.setStyleSheet("background-color: #c41e3a;")
            print(f"Recording started: {filename}")
        else:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self.record_btn.setText("🔴 Start Recording")
            self.record_btn.setStyleSheet("")
            print("Recording stopped.")

    def update_confidence(self, value):
        """Update confidence threshold"""
        self.conf_threshold = value / 100
        self.conf_label.setText(f"{self.conf_threshold:.2f}")

    def switch_model(self, model_name):
        """Switch YOLO model"""
        global model
        model = YOLO(YOLO_MODELS[model_name])
        print(f"Switched to model: {model_name}")

    def switch_camera(self, index):
        """Switch camera source"""
        camera_source = self.camera_combo.itemData(index)
        if camera_source is None:
            camera_source = 0
        
        if self.recording:
            self.toggle_recording()
        
        self.init_camera(camera_source)
        if self.cap is not None and self.cap.isOpened():
            print(f"Switched to camera: {camera_source}")
        else:
            print(f"Error switching camera: {camera_source}")

    def toggle_feature(self, feature):
        """Toggle visualization features"""
        if feature == 'trails':
            self.show_trails = self.trails_check.isChecked()
        elif feature == 'heatmap':
            self.show_heatmap = self.heatmap_check.isChecked()
            if not self.show_heatmap:
                self.heatmap = np.zeros((self.height, self.width), dtype=np.float32)
        elif feature == 'zones':
            self.show_zones = self.zones_check.isChecked()

    def clear_zones(self):
        """Clear all drawn zones"""
        self.zone_manager.clear_zones()
        self.zone_count_label.setText("Zones: 0 | Objects in zones: 0")
        print("All zones cleared")

    def add_alert(self, class_name):
        """Add class to alert monitoring"""
        if class_name and class_name != "Select class...":
            self.alert_manager.add_alert_class(class_name)
            active = ", ".join(self.alert_manager.alert_classes)
            self.active_alerts_label.setText(f"Active alerts: {active}")
            print(f"Alert added for: {class_name}")

    def export_report(self):
        """Export analytics report"""
        filename = f"reports/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.analytics.export_report(filename)

    def toggle_filter(self):
        """Toggle class filtering"""
        self.filter_enabled = self.filter_check.isChecked()
        if self.filter_enabled and not self.filtered_classes:
            # Default filter to common classes
            self.filtered_classes = {"person", "car", "truck", "bus", "bicycle", "motorcycle"}
            self.filter_info.setText(f"Showing: {', '.join(sorted(self.filtered_classes))}")
        elif not self.filter_enabled:
            self.filter_info.setText("No filters active")

    def closeEvent(self, event):
        """Clean up on application close"""
        self.running = False
        if self.recording and self.video_writer is not None:
            self.video_writer.release()
        if self.cap is not None:
            self.cap.release()
        event.accept()


# ----------------------------
# Clickable Label for Zone Drawing
# ----------------------------
class ClickableLabel(QLabel):
    """QLabel that emits signals on mouse events"""
    from PyQt6.QtCore import pyqtSignal
    mouse_pressed = pyqtSignal(int, int)
    mouse_double_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent
    
    def mousePressEvent(self, event):
        """Handle mouse press"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.mouse_pressed.emit(event.pos().x(), event.pos().y())
    
    def mouseDoubleClickEvent(self, event):
        """Handle double click"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.mouse_double_clicked.emit()
            if hasattr(self.parent_widget, 'on_image_double_click'):
                self.parent_widget.on_image_double_click()


# ----------------------------
# Main Application Entry
# ----------------------------
def main():
    app = QApplication(sys.argv)
    
    # Set application-wide font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    widget = VideoCaptureWidget()
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()