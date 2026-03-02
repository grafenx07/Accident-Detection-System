# 🚗 YOLOv8 Vehicle Crash Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Latest-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active-success)

*An intelligent real-time vehicle accident detection system powered by YOLOv8 deep learning model with instant WhatsApp notifications.*

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Configuration](#%EF%B8%8F-configuration) • [Screenshots](#-screenshots)

</div>

---

## 📋 Overview

This advanced Computer Vision system leverages YOLOv8 object detection to monitor live camera feeds and automatically detect vehicle accidents in real-time. Upon detection, it immediately sends WhatsApp alerts via Twilio, saves evidence images, and maintains a comprehensive detection log.

### 🎯 Key Highlights

- **Real-time Detection**: Processes live video feeds with high accuracy
- **Desktop GUI**: Modern interface built with CustomTkinter
- **Instant Alerts**: WhatsApp notifications via Twilio integration
- **Evidence Management**: Automatic saving of accident frames with timestamps
- **Multi-Camera Support**: Monitor multiple camera sources simultaneously
- **Detection History**: Complete log of all detected incidents with location tracking
- **Configurable Confidence Threshold**: Adjust detection sensitivity

---

## ✨ Features

### 🖥️ Desktop Application (main.py)

- **Modern Dark-themed GUI** built with CustomTkinter
- **Dashboard**: Live stats — total detections, active camera status, recent incident list
- **Real-time video processing** with live feed display
- **Media Upload & Analysis**: Upload images or videos for offline accident detection
- **Configurable detection settings**:
  - Adjustable confidence threshold (0-100%)
  - Camera source selection (Webcam/External)
  - Accident frame confirmation threshold
- **WhatsApp Notifications**:
  - Automatic alerts with location and timestamp
  - Test notification feature
  - Configurable cooldown period
- **Camera Location Management**: Assign names to different camera feeds
- **Evidence Storage**: Automatically saves accident frames and video clips with metadata
- **Detection History Viewer**: Browse all detected incidents with images
- **Export Logs**: Export detection data to CSV format

### 🌐 Web Application (app.py)

- **Streamlit-powered web UI** — run in any browser, no installation required beyond Python
- Upload and analyze video/image files directly from the browser
- Adjustable confidence slider, camera location input, enable/disable notifications
- Live detection log and statistics panel
- Evidence saving from the browser interface

### 🧠 AI Model

- **YOLOv8 Architecture**: State-of-the-art object detection
- **Custom Trained Model**: Specialized for accident detection
- **High Accuracy**: Optimized for various accident scenarios
- **Fast Inference**: Real-time processing capabilities

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Webcam or IP camera
- Twilio account (for WhatsApp notifications)

### Step 1: Clone the Repository

```bash
git clone https://github.com/grafenx07/Accident-Detection-System.git
cd Accident-Detection-System
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Model

**Important**: The model file is not included in the repository due to its size.

See [MODEL_DOWNLOAD.md](MODEL_DOWNLOAD.md) for instructions on:
- Training your own model
- Using pre-trained YOLOv8 models
- Model placement

The `best.pt` file must be in the root directory before running the application.

---

## 🚀 Usage

### Desktop Application

```bash
python main.py
```

**Getting Started:**

1. **Dashboard**: On launch, view live detection stats and recent incidents.

2. **Live Detection**:
   - Go to "Live Detection" tab and click **▶️ Start Detection**
   - Monitor the live camera feed in real-time

3. **Media Upload** (New):
   - Go to "Media Upload" tab
   - Browse and select an image or video file
   - Click **Analyze** — results and evidence are saved automatically

4. **Configure Detection Settings**:
   - Navigate to "Settings" tab
   - Adjust confidence threshold (default: 0.5)
   - Select camera source (Webcam/External)
   - Enable/disable evidence saving

5. **Setup WhatsApp Notifications** (Optional):
   - Enable WhatsApp notifications in Settings
   - Enter your Twilio credentials:
     - Account SID
     - Auth Token
     - Twilio phone number
     - Recipient phone number
   - Click "Save Notification Settings"
   - Test with "Test WhatsApp Notification" button

6. **Configure Camera Locations**:
   - Assign names to your camera feeds in Settings
   - Click "Save Location Settings"

7. **View Detection History**:
   - Go to "History" tab to browse all detected incidents and evidence images

---

### Web Application (Streamlit)

```bash
streamlit run app.py
```

- Opens automatically in your browser at `http://localhost:8501`
- Use the sidebar to configure confidence, camera location, and notifications
- Upload a video/image or start a live feed to begin detection

---

## ⚙️ Configuration

### Twilio WhatsApp Setup

1. **Create Twilio Account**:
   - Visit [Twilio](https://www.twilio.com/)
   - Sign up for a free account

2. **Enable WhatsApp Sandbox**:
   - Go to Twilio Console → Messaging → Try it out → Send a WhatsApp message
   - Follow instructions to connect your WhatsApp

3. **Get Credentials**:
   - Account SID: Found in Twilio Console Dashboard
   - Auth Token: Found in Twilio Console Dashboard
   - Twilio Number: Your Twilio WhatsApp-enabled number

4. **Configure in Application**:
   - Open the app
   - Navigate to "Notification Settings"
   - Enter your credentials
   - Save and test

### Detection Parameters

Edit these variables in `main.py` for customization:

```python
confidence_threshold = 0.5          # Detection confidence (0-1)
notification_cooldown = 10          # Seconds between notifications
accident_frames_threshold = 5       # Frames to confirm accident
```

### Camera Configuration

- **Webcam**: Set to `0` (default)
- **External Camera**: Set to `1` or camera URL
- **IP Camera**: Use RTSP URL format

---

## 📁 Project Structure

```
Accident-Detection-System/
│
├── main.py                          # Desktop application (CustomTkinter)
├── app.py                           # Web application (Streamlit)
├── main_backup.py                   # Backup of previous main version
├── train.py                         # Model training script
├── download_and_merge_datasets.py   # Dataset download & merge utility
├── best.pt                          # YOLOv8 trained model (not in repo — download separately)
├── coco1.txt                        # Class labels
├── data.yaml                        # YOLO dataset configuration
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore file
│
└── accident_evidence/               # Saved accident frames/clips (generated at runtime)
```

---

## 📦 Dependencies

### Core Libraries

- **ultralytics** - YOLOv8 implementation
- **opencv-python** - Computer vision operations
- **customtkinter** - Modern GUI framework (desktop app)
- **streamlit** - Web application framework
- **pandas** - Data manipulation
- **Pillow** - Image processing
- **twilio** - WhatsApp messaging
- **cvzone** - Computer vision utilities

See `requirements.txt` for complete list.

---

## 🔒 Security & Privacy

- **Credentials**: Never commit sensitive data (Twilio credentials)
- **Evidence Storage**: Accident images saved locally only
- **Network Security**: Use HTTPS for IP camera streams
- **Data Privacy**: No data transmitted except WhatsApp alerts

---

## 🎨 Screenshots

### Desktop Application

- Modern dark-themed interface
- Real-time video feed with bounding boxes
- Live detection status
- Configurable settings tabs

---

## 🚦 How It Works

1. **Video Capture**: System captures frames from camera/video source
2. **Preprocessing**: Frames resized and prepared for model input
3. **Detection**: YOLOv8 model analyzes each frame for accidents
4. **Confirmation**: Multiple consecutive detections required to confirm
5. **Alert**: WhatsApp notification sent with location and timestamp
6. **Evidence**: Frame saved with metadata to accident_evidence folder
7. **Logging**: Incident recorded in detection history

---

## 🔧 Troubleshooting

### Common Issues

**Camera Not Opening:**
- Ensure camera is connected and not in use by another application
- Try different camera IDs (0, 1, 2)
- Check camera permissions

**WhatsApp Notifications Not Working:**
- Verify Twilio credentials are correct
- Ensure WhatsApp sandbox is activated
- Check phone number format (+country code)
- Verify internet connection

**Low Detection Accuracy:**
- Adjust confidence threshold
- Ensure good lighting conditions
- Check camera angle and position
- Verify model file (best.pt) is present

**Performance Issues:**
- Reduce frame processing rate
- Lower camera resolution
- Close unnecessary applications
- Ensure sufficient RAM available

---

## 📈 Future Enhancements

- [ ] Multi-threaded camera processing
- [ ] Cloud storage integration for evidence
- [ ] SMS alert option
- [ ] Email notifications
- [ ] Mobile app version
- [ ] Advanced analytics dashboard
- [ ] Integration with traffic management systems
- [ ] Vehicle license plate recognition
- [ ] Severity classification

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Authors

- **grafenx07** - [GitHub](https://github.com/grafenx07)

---

## 🙏 Acknowledgments

- **Ultralytics** for YOLOv8 framework
- **Twilio** for messaging API
- **OpenCV** community
- All contributors and testers

---

## 📞 Support

For issues, questions, or suggestions:

- Open an issue on GitHub
- Contact: [GitHub Profile](https://github.com/grafenx07)

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

<div align="center">

**Made with ❤️ using Python and YOLOv8**

[Back to Top](#-yolov8-vehicle-crash-detection-system)

</div>
