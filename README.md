**TorchObjectDetection: Real-time Object Detection with YOLOv8 and PyQt6**
===========================================================

### Project Title
A real-time object detection system powered by YOLOv8, running on a PyQt6 GUI application.

### Description
This project is my take on integrating the mighty YOLOv8 model with the Pythonic goodness of PyQt6. I built this to demonstrate how easily you can create a robust and efficient object detection system, perfect for applications where real-time tracking is crucial. With TorchObjectDetection, you can:

* Load pre-trained YOLOv8 models (or use your own!)
* Capture video feed from any webcam
* Detect objects in real-time with high accuracy

### Features
Some of the cooler features I'm proud to showcase include:

* **Real-time object detection**: Update frames at an impressive 30fps, giving you smooth tracking and response.
* **Multiple model support**: Choose from various YOLOv8 variants (e.g., yolov10m.pt, yolov9c.pt) or use your own custom models!
* **Configurable webcam resolution**: Set the capture resolution to suit your needs.
* **Interactive GUI**: A clean and intuitive PyQt6 interface lets you monitor the video feed and object detections in real-time.

### Installation
Before we dive into usage, make sure you have:

1. Python 3.x (preferably the latest version)
2. `torch` and `pytorchcv` installed (`pip install torch torchvision pytorchcv`)
3. `ultralytics` library for YOLOv8 models (`git clone https://github.com/ultralytics/yolov5.git && pip install .`)

### Usage
To get started, simply run the main script (`python main.py`) and:

1. Connect your webcam to your machine.
2. Set the `camera_index` variable in `main.py` to match your camera's index (e.g., 0 for a USB camera).
3. Run the application using PyQt6.

```markdown
# Main Application
if __name__ == '__main__':
    app = QApplication(sys.argv)

    # ... initialization code ...

    video_capture_widget = VideoCaptureWidget(camera_index=1)
    video_capture_widget.show()

    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("Exiting gracefully...")
        video_capture_widget.close()
```

### Contributing
Contributions are welcome! If you'd like to enhance this project, please open an issue or PR with your suggested changes.

### License
TorchObjectDetection is licensed under the permissive MIT license. You're free to use, modify, and distribute this code as you see fit.

### Tags/Keywords

* `real-time object detection`
* `yolov8`
* `pyqt6`
* `python object detection`
* `computer vision`
* `AI-powered tracking`

Get ready to unleash the power of YOLOv8 on your next project!