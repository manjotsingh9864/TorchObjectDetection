import sys
import cv2
import torch
import threading
from ultralytics import YOLO
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt6.QtGui import QImage, QPixmap

# Load the pre-trained YOLOv8 model
model = YOLO('yolov10m.pt')  # Use the YOLOv8 model (you can choose a different variant)

# Function to set the highest resolution supported by the webcam
def set_max_resolution(cap):
    common_resolutions = [
        #(3840, 2160),  # 4K
        #(2560, 1440),  # 1440p
        #(1920, 1080),  # 1080p
        (1280, 720),   # 720p
        (640, 480),    # 480p
    ]
    for width, height in common_resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if (actual_width, actual_height) == (width, height):
            print(f"Resolution set to: {width}x{height}")
            return width, height
    return 640, 480

# Define the main window class
class VideoCaptureWidget(QWidget):
    def __init__(self, camera_index=1):
        super().__init__()
        self.setWindowTitle("Object Detection with PyQt6 and YOLOv8")
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_label)
        self.setLayout(self.layout)

        self.cap = cv2.VideoCapture(camera_index)  # Open the specified webcam
        self.width, self.height = set_max_resolution(self.cap)
        self.image_label.resize(self.width, self.height)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # Update frame every 33ms for ~30fps

        self.lock = threading.Lock()
        self.frame = None
        self.results = None

        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()

    def capture_frames(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
                self.detect_objects(frame)

    def detect_objects(self, frame):
        results = model(frame)
        with self.lock:
            self.results = results

    def update_frame(self):
        with self.lock:
            frame = self.frame
            results = self.results

        if frame is None or results is None:
            return

        for result in results[0].boxes:
            box = result.xyxy[0].cpu().numpy().astype(int)
            score = float(result.conf.cpu().numpy())
            label = int(result.cls.cpu().numpy())
            label_name = model.names[label]  # Convert label to class name
            if score >= 0.5:
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                text = f"{label_name}: {score:.2f}"
                cv2.putText(frame, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        if hasattr(self, 'cap'):
            self.update_frame()

    def closeEvent(self, event):
        self.cap.release()
        super().closeEvent(event)

# Main application
if __name__ == '__main__':
    app = QApplication(sys.argv)

    camera_index = 1
    video_capture_widget = VideoCaptureWidget(camera_index=camera_index)
    video_capture_widget.show()

    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("Exiting gracefully...")
        video_capture_widget.close()
