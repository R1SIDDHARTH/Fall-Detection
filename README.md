# Fall Detection System

## Overview
The Fall Detection System is a Python-based project designed to detect falls in real-time or from video clips using computer vision and machine learning techniques. It integrates YOLO for object detection, MediaPipe for pose estimation, and various alert mechanisms (SMS, voice alerts, and emergency calls) to ensure timely responses to detected falls. The system is modular, with three main Python scripts tailored for different functionalities: real-time fall detection, video clip analysis, and alert handling.

## Project Structure
The project contains three main Python files:
1. **simple-fall.py**: Implements real-time fall detection using a webcam, with YOLO for person detection and MediaPipe for pose estimation. It includes SMS alerts, voice prompts, and emergency call functionalities.
2. **fall-(clip).py**: Processes pre-recorded video clips to detect falls using MediaPipe's pose estimation. It supports batch processing of multiple videos and generates a detailed report.
3. **fall-aleart.py**: Focuses on real-time fall detection with simplified detection logic and enhanced alert mechanisms, including SMS, emergency calls, and audio playback.

## Features
- **Real-Time Fall Detection**: Monitors live video feed from a webcam to detect falls using YOLO and MediaPipe.
- **Video Clip Analysis**: Analyzes pre-recorded videos to identify fall events and generates a report.
- **Alert System**: Sends SMS alerts via Twilio, plays audio prompts, and initiates emergency calls if no response is received.
- **Pose Estimation**: Uses MediaPipe to track body keypoints and detect abnormal postures or lying positions indicative of a fall.
- **Voice Interaction**: Listens for voice responses to confirm the user's status after a fall is detected.
- **Recording**: Automatically records video footage when a fall is detected for later review.
- **Batch Processing**: Processes multiple video files in a specified range with customizable start and end percentages.

## Requirements
To run the project, install the required Python packages listed in `requirements.txt`. The main dependencies include:
- OpenCV (`cv2`)
- NumPy
- Ultralytics YOLO (`ultralytics`)
- MediaPipe (`mediapipe`)
- Twilio (`twilio`)
- SpeechRecognition (`speech_recognition`)
- gTTS (`gtts`)
- Pygame (`pygame`)
- cvzone

Install them using:
```bash
pip install -r requirements.txt
```

Additionally, you need:
- A Twilio account with valid `ACCOUNT_SID`, `AUTH_TOKEN`, and phone number for SMS and call alerts.
- A webcam for real-time detection (for `simple-fall.py` and `fall-aleart.py`).
- Audio files (`Are-u-ok.mp3`, `Emergency.mp3`) for `fall-aleart.py` or a system capable of generating beeps.

## Setup
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd fall-deduction
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Twilio**:
   - Update `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_PHONE_NUMBER`, and `EMERGENCY_CONTACT` in `simple-fall.py` and `fall-aleart.py` with your Twilio credentials and contact numbers.

4. **Set File Paths**:
   - In `simple-fall.py`, update `Settings.RECORDING_PATH` to a valid directory for saving recordings.
   - In `fall-(clip).py`, ensure `video_dir` and `output_dir` paths are correct for input videos and output files.
   - In `fall-aleart.py`, verify the `audio_base_path` and ensure audio files exist, or the system will fall back to beeps.

5. **Download YOLO Model**:
   - For `simple-fall.py`, ensure the YOLOv8 model (`yolov8s.pt`) is available. It will be automatically downloaded by Ultralytics if not present.

## Usage
### 1. Real-Time Fall Detection (`simple-fall.py`)
Run the script to start real-time fall detection using a webcam:
```bash
python simple-fall.py
```
- The system monitors the video feed, detects falls, and triggers alerts (SMS, voice prompts, emergency calls).
- Press `q` to quit.

### 2. Video Clip Analysis (`fall-(clip).py`)
Analyze pre-recorded videos for fall detection:
```bash
python fall-(clip).py --video_dir <input_video_directory> --output_dir <output_directory> --start <start_video_number> --end <end_video_number> --start_percent <start_percent> --end_percent <end_percent>
```
- Example:
  ```bash
  python fall-(clip).py --video_dir "C:\Videos\Fall-Clip" --output_dir "C:\Videos\Analysed-Clip" --start 1 --end 31 --start_percent 0 --end_percent 100
  ```
- The script processes videos, saves analyzed clips, and generates a report (`fall_detection_report.txt`).
- Use `--no_preview` to disable the preview window, and `--display_width`/`--display_height` to resize the display.

### 3. Real-Time Fall Detection with Alerts (`fall-aleart.py`)
Run the script for real-time detection with enhanced alert handling:
```bash
python fall-aleart.py
```
- The system monitors the webcam feed, detects falls, and triggers alerts (SMS, audio, emergency calls).
- Keyboard controls:
  - `q`: Quit
  - `f`: Force fall detection
  - `a`: Test audio playback
  - `c`: Test emergency call
  - `s`: Test SMS
  - `r`: Reset system

## Output
- **Real-Time Detection**: Displays a live feed with annotations (e.g., "FALL DETECTED!", skeleton overlay, status indicators).
- **Video Analysis**: Saves analyzed videos with annotations and a report detailing fall events, frames, and timestamps.
- **Alerts**: Sends SMS, plays audio prompts, and initiates emergency calls when falls are detected and no response is received.
- **Recordings**: Saves video clips of fall events to the specified output directory.

## Limitations
- **Real-Time Detection**: Requires a stable webcam feed and sufficient lighting for accurate pose estimation.
- **Video Analysis**: Performance depends on video quality and frame rate. Low-quality videos may lead to missed detections.
- **Alerts**: Twilio services require an active account and internet connection. Voice recognition may fail in noisy environments.
- **False Positives**: The system may misinterpret certain movements (e.g., lying down intentionally) as falls. Fine-tuning thresholds can help.
- **Audio Files**: `fall-aleart.py` relies on specific audio files; missing files result in fallback beeps.

## Future Improvements
- Enhance fall detection accuracy by combining YOLO and MediaPipe with additional machine learning models.
- Implement a user interface for easier configuration and monitoring.
- Add support for multiple cameras or IP streams.
- Improve voice recognition robustness in noisy environments.
- Integrate with smart home devices for automated alerts.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For questions or support, contact the project maintainer at [siddharth.r.college@gmail.com](mailto:siddharth.r.college@gmail.com).