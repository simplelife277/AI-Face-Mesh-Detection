# 🤖 AI Face Mesh Detection Using Python

A real-time AI-powered Face Mesh Detection project built using **Python, OpenCV, and MediaPipe Tasks**. This project detects facial landmarks through a webcam and draws a mesh over the detected face.

## 🚀 Features

- Real-time face landmark detection
- Webcam integration using OpenCV
- Face mesh tessellation visualization
- Face detection status display
- FPS (Frames Per Second) monitoring
- Automatic webcam detection
- ESC key to exit the application
- Error handling for missing model files and webcam issues

## 🛠️ Technologies Used

- Python 3.14.4
- OpenCV
- MediaPipe Tasks
- NumPy

## 📂 Project Structure

```text
Face-Mesh/
│
├── .venu/
├── main.py
├── face_landmarker.task
├── requirements.txt
└── README.md
```

## 📦 Installation

### 1. Clone the Repository

```bash
https://github.com/simplelife277/AI-Face-Mesh-Detection
```

### 2. Navigate to the Project Folder

```bash
cd Face-Mesh
```

### 3. Create a Virtual Environment

```bash
python -m venv .venu
```

### 4. Activate the Virtual Environment

Windows PowerShell:

```powershell
.\.venu\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## 📥 Download MediaPipe Model

Download the Face Landmarker model:

[Download Face Landmarker Model](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task)

Place `face_landmarker.task` in the main project folder.

## ▶️ Run the Project

```bash
python main.py
```

Your webcam will open and detect facial landmarks in real time.

Press **ESC** to close the application.

## 🧠 How It Works

1. OpenCV captures video from the webcam.
2. Each video frame is converted from BGR to RGB.
3. MediaPipe Face Landmarker detects facial landmarks.
4. Facial landmark connections are drawn to create a face mesh.
5. The application displays face detection status and FPS.
6. Pressing ESC stops the program and releases the webcam.

## 🎯 Learning Outcomes

- Understanding computer vision fundamentals
- Working with real-time webcam input
- Using MediaPipe Tasks for AI-based detection
- Understanding facial landmark detection
- Drawing facial mesh connections using OpenCV
- Managing Python packages and virtual environments

## 🔮 Future Improvements

- Eye tracking
- Face expression detection
- Face distance estimation
- Face recognition integration
- Hand gesture detection

## 👨‍💻 Author

**Saravanan T.**

Python Developer | Backend Development Enthusiast

## ⭐ Support

If you find this project useful, please give this repository a star!

---

**#Python #OpenCV #MediaPipe #FaceMesh #ComputerVision #ArtificialIntelligence #MachineLearning #PythonProjects**
