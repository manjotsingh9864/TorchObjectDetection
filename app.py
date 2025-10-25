import sys
import cv2
import threading
import time
from ultralytics import YOLO
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QApplication, QLabel, QVBoxLayout, QWidget, QPushButton, QHBoxLayout,
    QComboBox, QSlider, QGridLayout
)
from PyQt6.QtGui import QImage, QPixmap, QFont
import os
import numpy as np

# ----------------------------
# YOLO Models Configuration
# ----------------------------
YOLO_MODELS = {
    "YOLOv8n": "yolov8n.pt",
    "YOLOv8m": "yolov8m.pt",
    "YOLOv8l": "yolov8l.pt"
}
current_model_name = "YOLOv8m"
model = None  # Initialize later after camera selection

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
# Main GUI Class
# ----------------------------
class VideoCaptureWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8 Real-Time Object Detection - Portfolio App")
        self.setGeometry(50, 50, 1280, 720)

        # Initialize camera index
        self.camera_index = 0
        self.cap = None
        self.width = 0
        self.height = 0
        self.camera_error_label = QLabel("", self)
        self.camera_error_label.setFont(QFont("Arial", 12))
        self.camera_error_label.setStyleSheet("color: red;")
        self.camera_error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Initialize YOLO model here after camera selection
        global model
        model = YOLO(YOLO_MODELS[current_model_name])

        # ---------------- GUI Elements ----------------
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Stats
        self.fps_label = QLabel("FPS: 0", self)
        self.fps_label.setFont(QFont("Arial", 12))
        self.count_label = QLabel("Objects: 0", self)
        self.count_label.setFont(QFont("Arial", 12))

        # Buttons
        self.snapshot_btn = QPushButton("Save Snapshot")
        self.snapshot_btn.clicked.connect(self.save_snapshot)
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.recording = False
        self.video_writer = None

        # Confidence slider
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setMinimum(1)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(25)
        self.conf_slider.valueChanged.connect(self.update_confidence)
        self.conf_label = QLabel("Confidence: 0.25")

        # YOLO model combo
        self.model_combo = QComboBox()
        self.model_combo.addItems(YOLO_MODELS.keys())
        self.model_combo.setCurrentText(current_model_name)
        self.model_combo.currentTextChanged.connect(self.switch_model)

        # Camera selection combo box
        self.camera_combo = QComboBox()
        # On macOS, AVFoundation backend can use device name string for iPhone (Continuity Camera)
        # Use integer 0 for MacBook camera, string "iPhone" for iPhone camera to distinguish
        self.camera_combo.addItem("MacBook Camera", 0)
        self.camera_combo.addItem("iPhone Camera (Continuity Camera)", "iPhone")
        self.camera_combo.setCurrentIndex(0)
        self.camera_combo.currentIndexChanged.connect(self.switch_camera)

        # Layouts
        control_layout = QGridLayout()
        control_layout.addWidget(self.fps_label, 0, 0)
        control_layout.addWidget(self.count_label, 0, 1)
        control_layout.addWidget(self.snapshot_btn, 0, 2)
        control_layout.addWidget(self.record_btn, 0, 3)
        control_layout.addWidget(QLabel("Confidence:"), 1, 0)
        control_layout.addWidget(self.conf_slider, 1, 1, 1, 2)
        control_layout.addWidget(self.conf_label, 1, 3)
        control_layout.addWidget(QLabel("YOLO Model:"), 2, 0)
        control_layout.addWidget(self.model_combo, 2, 1)
        control_layout.addWidget(QLabel("Camera:"), 3, 0)
        control_layout.addWidget(self.camera_combo, 3, 1)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.camera_error_label)
        self.layout.addLayout(control_layout)
        self.setLayout(self.layout)

        # ---------------- Threading & FPS ----------------
        self.lock = threading.Lock()
        self.frame = None
        self.results = None
        self.last_time = time.time()
        self.fps = 0
        self.conf_threshold = 0.25
        self.running = True

        # Initialize camera
        self.init_camera(self.camera_combo.currentData())

        # Start capture thread
        self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        self.capture_thread.start()

        # Timer for GUI update
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # ~30 FPS

    def init_camera(self, camera_source):
        # Release previous capture if exists
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.camera_error_label.setText("")
        try:
            if sys.platform == "darwin":
                # MacBook Camera uses index 0 with CAP_AVFOUNDATION
                # iPhone Camera uses device name string "iPhone" with CAP_AVFOUNDATION
                if isinstance(camera_source, str):
                    self.cap = cv2.VideoCapture(camera_source, cv2.CAP_AVFOUNDATION)
                else:
                    self.cap = cv2.VideoCapture(camera_source, cv2.CAP_AVFOUNDATION)
            else:
                # Default to index for other platforms
                self.cap = cv2.VideoCapture(camera_source)
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = None
                error_message = "Cannot open camera. Check camera permissions or connection."
                print(error_message)
                self.camera_error_label.setText(error_message)
                self.width, self.height = 640, 480  # fallback resolution
                return
            self.width, self.height = set_camera_resolution(self.cap)
            self.camera_error_label.setText("")
        except Exception as e:
            if self.cap is not None:
                self.cap.release()
            self.cap = None
            error_message = f"Camera initialization error: {e}"
            print(error_message)
            self.camera_error_label.setText(error_message)
            self.width, self.height = 640, 480  # fallback resolution

    # ---------------- Frame Capture ----------------
    def capture_frames(self):
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
                time.sleep(0.01)  # Avoid busy loop if no frame

    # ---------------- YOLO Detection ----------------
    def detect_objects(self, frame):
        global model
        results = model(frame, device="mps", conf=self.conf_threshold)
        with self.lock:
            self.results = results

    # ---------------- Update GUI ----------------
    def update_frame(self):
        with self.lock:
            frame = self.frame.copy() if self.frame is not None else None
            results = self.results

        if frame is None or results is None:
            return

        # Object counting per class
        object_count = 0
        class_counts = {}
        for result in results[0].boxes:
            box = result.xyxy[0].cpu().numpy().astype(int)
            score = float(result.conf.cpu().numpy())
            label = int(result.cls.cpu().numpy())
            label_name = model.names[label]
            if score >= self.conf_threshold:
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"{label_name}:{score:.2f}", (box[0], max(box[1]-10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                object_count += 1
                class_counts[label_name] = class_counts.get(label_name, 0) + 1

        # FPS calculation
        current_time = time.time()
        dt = current_time - self.last_time
        if dt > 0:
            self.fps = 1 / dt
        self.last_time = current_time
        self.fps_label.setText(f"FPS: {self.fps:.2f}")
        if class_counts:
            counts_text = ", ".join([f"{k}:{v}" for k,v in class_counts.items()])
            self.count_label.setText(f"Objects: {object_count} | {counts_text}")
        else:
            self.count_label.setText(f"Objects: {object_count}")

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

    # ---------------- Snapshot ----------------
    def save_snapshot(self):
        with self.lock:
            if self.frame is not None:
                os.makedirs("snapshots", exist_ok=True)
                filename = f"snapshots/snapshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(filename, self.frame)
                print(f"Snapshot saved: {filename}")

    # ---------------- Recording ----------------
    def toggle_recording(self):
        self.recording = not self.recording
        if self.recording:
            os.makedirs("videos", exist_ok=True)
            filename = f"videos/recording_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 compatible with QuickTime
            self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (self.width, self.height))
            self.record_btn.setText("Stop Recording")
            print(f"Recording started: {filename}")
        else:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self.record_btn.setText("Start Recording")
            print("Recording stopped.")

    # ---------------- Confidence Slider ----------------
    def update_confidence(self, value):
        self.conf_threshold = value / 100
        self.conf_label.setText(f"Confidence: {self.conf_threshold:.2f}")

    # ---------------- YOLO Model Switch ----------------
    def switch_model(self, model_name):
        global model
        model = YOLO(YOLO_MODELS[model_name])
        print(f"Switched to model: {model_name}")

    # ---------------- Camera Switch ----------------
    def switch_camera(self, index):
        camera_source = self.camera_combo.itemData(index)
        if camera_source is None:
            camera_source = 0
        # Stop recording if active
        if self.recording:
            self.toggle_recording()
        # Reinitialize camera
        self.init_camera(camera_source)
        if self.cap is not None and self.cap.isOpened():
            print(f"Switched to camera: {camera_source}")
        else:
            print(f"Error switching camera: Cannot open camera {camera_source}")

    # ---------------- Clean Exit ----------------
    def closeEvent(self, event):
        self.running = False
        if self.recording and self.video_writer is not None:
            self.video_writer.release()
        if self.cap is not None:
            self.cap.release()
        event.accept()


# ----------------------------
# Main Application Entry
# ----------------------------
def main():
    app = QApplication(sys.argv)
    widget = VideoCaptureWidget()
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()