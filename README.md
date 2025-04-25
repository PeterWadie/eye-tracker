# Eye Tracker Software

Real-time eye-tracking with blink detection, gaze estimation, and fatigue alerts.

## Prerequisites

- **Python 3.10+**
- **Docker (19.03+)** _(optional, for cross-platform portability)_
- A **USB webcam** or built-in camera

## Repository Structure

```text
eye-tracker/
├── config.json
├── Dockerfile
├── requirements.txt
├── main.py
├── utils.py
├── detectors.py
├── models/
│   ├── deploy.prototxt
│   ├── res10_300x300_ssd_iter_140000_fp16.caffemodel
│   └── shape_predictor_68_face_landmarks.dat
└── README.md
```

## Configuration

Edit `config.json` to tweak parameters:

```json
{
  "camera_index": 0,
  "width": 640,
  "height": 480,
  "fps": 30,
  "log_downsample_rate": 2,
  "ear_blink_threshold": 0.2,
  "ear_closed_secs": 3.0,
  "shape_predictor_path": "models/shape_predictor_68_face_landmarks.dat",
  "dnn_model": {
    "prototxt": "models/deploy.prototxt",
    "caffemodel": "models/res10_300x300_ssd_iter_140000_fp16.caffemodel",
    "conf_threshold": 0.5
  }
}
```

## Download Pre-trained Models

Create the `models/` directory and download the required files:

```bash
mkdir -p models

# 1) Dlib 68-point facial landmark predictor
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 \
     -O models/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d models/shape_predictor_68_face_landmarks.dat.bz2

# 2) OpenCV DNN face detector definition (.prototxt)
wget https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt \
     -O models/deploy.prototxt

# 3) OpenCV DNN face detector weights (FP16 Caffe model)
wget https://raw.githubusercontent.com/spmallick/learnopencv/master/FaceDetectionComparison/models/res10_300x300_ssd_iter_140000_fp16.caffemodel \
     -O models/res10_300x300_ssd_iter_140000_fp16.caffemodel
```

## Local (Native) Setup & Run

1. **Create & activate** a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install** dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Run** the tracker:
   ```bash
   python main.py
   ```

You should see your webcam feed annotated with EAR, gaze direction, and alarms.

## Docker Build & Run

> **Note:** macOS Docker cannot access the host camera by default. For full Docker webcam support, use a Linux host or VM.

1. **Build** the image:
   ```bash
   docker build -t eye-tracker:latest .
   ```

2. **Run** the container:
   ```bash
   docker run -it \
     --device /dev/video0:/dev/video0 \
     -v "$(pwd)/models":/app/models \
     -v "$(pwd)/config.json":/app/config.json \
     eye-tracker:latest
   ```

The container will launch `main.py` and display the video stream (on Linux hosts).

## Troubleshooting

- **ImportError / Module not found**  
  Ensure you’re in the virtualenv (native) or that `requirements.txt` lists `opencv-python`, `dlib`, and `numpy`.

- **Camera not detected in Docker (macOS)**  
  Use the native Python instructions, or test in a Linux VM with `/dev/video0` support.

- **Performance issues**  
  - Lower resolution or FPS in `config.json`.  
  - On M1 Mac, try up to 1280×720 @30 FPS if performance allows.

---
