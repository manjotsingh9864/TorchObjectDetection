# 🚀 TorchObjectDetection — YOLOv8 Vision Suite by Manjot Singh

### 👨‍💻 Author: [Manjot Singh](https://www.linkedin.com/in/manjot-singh-ds/)
📧 Email: [singhtmanjot@gmail.com](mailto:singhtmanjot@gmail.com)

---

## 🧾 Overview

**TorchObjectDetection** is a real-time computer vision application built with **YOLOv8**, **OpenCV**, and **PyQt6**.  
It allows you to perform live **object detection**, **zone tracking**, **analytics**, **recording**, and **heatmap visualization** through a clean, modern interface.

This app combines **deep learning**, **data analytics**, and **UI design** — making it ideal for:
- 🔒 Security and surveillance
- 🏪 Smart retail analytics
- 🚗 Traffic and crowd monitoring
- 🎓 Educational demonstrations of real-time object detection

---

## ✨ Key Features

✅ **Real-Time YOLOv8 Detection** — Detect multiple object classes live using webcam or IP camera.  
✅ **Zone Drawing Tool** — Draw and monitor Regions of Interest (ROIs) to count objects inside.  
✅ **Dynamic Model Switching** — Instantly switch between YOLOv8n, YOLOv8m, and YOLOv8l models.  
✅ **Heatmap Visualization** — View live motion activity heatmaps.  
✅ **Analytics Dashboard** — View detection stats, peak counts, and moving averages.  
✅ **Class-Based Alerts** — Trigger notifications or sounds for specific detected classes.  
✅ **Snapshots & Recording** — Capture snapshots and record high-quality videos.  
✅ **Autosave Session Reports** — Automatically exports JSON reports every 5 minutes.  
✅ **Keyboard Shortcuts:**

| Shortcut | Action |
|-----------|--------|
| `R` | Start/Stop Recording |
| `S` | Save Snapshot |
| `P` | Pause/Resume Feed |
| `Q` | Quit Application |

---

## 🧰 Tech Stack

| Category | Tools |
|-----------|--------|
| **Deep Learning** | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) |
| **Computer Vision** | OpenCV |
| **GUI Framework** | PyQt6 |
| **Visualization** | pyqtgraph |
| **Language** | Python 3.10+ |
| **Environment** | Virtual Environment (`yolov8-env`) |

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/TorchObjectDetection.git
cd TorchObjectDetection
```

### 2️⃣ Create a Virtual Environment
It’s highly recommended to use a virtual environment for isolation.

#### 🧪 Using `venv`:
```bash
python3 -m venv yolov8-env
source yolov8-env/bin/activate  # For Mac/Linux
yolov8-env\Scripts\activate     # For Windows
```

---

### 3️⃣ Install Dependencies
Make sure you’re inside your environment, then run:
```bash
pip install -r requirements.txt
```

If you don’t have one yet, install manually:
```bash
pip install ultralytics PyQt6 opencv-python pyqtgraph numpy
```

---

### 4️⃣ Run the Application
```bash
python3 app3.py
```

> 💡 **Note for macOS Users:**  
> If your webcam doesn’t start, go to  
> **System Settings → Privacy & Security → Camera** and allow Python access.

---

## 📁 Project Structure

```
TorchObjectDetection/
├── app3.py                 # Main YOLOv8 PyQt6 Application
├── videos/                 # Saved video recordings
├── snapshots/              # Saved snapshots
├── reports/                # Exported analytics reports
├── exports/                # Autosaved session data
├── yolov8-env/             # Virtual environment folder (not uploaded)
├── requirements.txt        # Python dependencies
└── README.md               # Documentation file
```

---

## 📈 Example Use Cases

- 🔒 **Security Monitoring** – Detect people entering restricted zones  
- 🏪 **Retail Analytics** – Count customers in defined areas  
- 🚗 **Traffic Analysis** – Track vehicles and congestion  
- 🎓 **Research / Education** – Demonstrate AI-powered object detection  

---

## 🧮 Report & Data Exports

Your app automatically exports detailed JSON reports like this:

```json
{
  "session_duration": 582.34,
  "total_detections": 1840,
  "class_distribution": {"person": 1600, "car": 240},
  "peak_detections": 14,
  "average_detections": 3.8
}
```

Reports are stored in the `reports/` or `exports/` directories, and include:
- Total detections  
- Detection averages  
- Peak detections  
- Class distribution  

---

## 📸 Screenshots (Optional)

You can include screenshots in your GitHub repo like:
```
![App Interface](assets/screenshot1.png)
![Zone Detection Example](assets/screenshot2.png)
```

---

## 🧑‍💻 Developer Info

👨‍💻 **Author:** [Manjot Singh](https://www.linkedin.com/in/manjot-singh-ds/)  
📧 **Email:** [singhtmanjot@gmail.com](mailto:singhtmanjot@gmail.com)  
🏷️ **Project Name:** TorchObjectDetection  
💻 **Virtual Environment:** `yolov8-env`

Feel free to connect for collaborations in **AI**, **Deep Learning**, or **Computer Vision** projects.

---

## ⚖️ License

This project is open-source under the **MIT License** — free to use, modify, and distribute with credit to the author.

---

## 🌟 Acknowledgments

Special thanks to:
- [Ultralytics](https://github.com/ultralytics) — YOLOv8 Framework  
- [OpenCV](https://opencv.org/) — Computer Vision Tools  
- [PyQt6](https://riverbankcomputing.com/software/pyqt/intro) — GUI Framework  

---

> 🏁 *Built with ❤️ by Manjot Singh — Merging AI, Vision, and Creativity.*