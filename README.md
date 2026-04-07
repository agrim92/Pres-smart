# Pres-smart

Gesture-controlled presentation assistant built with Python, OpenCV, MediaPipe, and PyAutoGUI.

## Files

- `main.py`: webcam loop, MediaPipe integration, slide control, and overlay updates
- `gesture_detection.py`: gesture classification and motion-based swipe detection
- `utils.py`: drawing helpers, finger-state logic, and shared utilities
- `requirements.txt`: Python dependencies

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Download the MediaPipe hand-landmarker model as `hand_landmarker.task` into the project folder, then run:

```powershell
python main.py
```

Press `q` in the webcam window to quit.
