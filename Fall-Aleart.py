import cv2
import mediapipe as mp
import numpy as np
import os
import time
from datetime import datetime
import threading
import pygame
import speech_recognition as sr
from twilio.rest import Client

# Twilio credentials
TWILIO_ACCOUNT_SID = "ACd0f3a2b5c4e1f7a8b9c0d1e2f3g4h5i6"
TWILIO_AUTH_TOKEN = " your_twilio_auth_token"
# Twilio phone number and emergency contact
TWILIO_PHONE_NUMBER = "+14155238886"
# Emergency contact number (replace with your own)
EMERGENCY_CONTACT = "+1234567890"

def send_emergency_sms():
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body="🚨 Alert: A fall has been detected. Please check in immediately.",
            from_=TWILIO_PHONE_NUMBER,
            to=EMERGENCY_CONTACT,
        )
        print("✅ Emergency SMS sent successfully! SID:", message.sid)
        return True
    except Exception as e:
        print(f"❌ Failed to send emergency SMS: {e}")
        return False


def make_emergency_call():
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        call = client.calls.create(
            twiml="<Response><Say>Emergency! A fall has been detected and the person is unresponsive. Please assist immediately.</Say></Response>",
            from_=TWILIO_PHONE_NUMBER,
            to=EMERGENCY_CONTACT,
        )
        print("✅ Emergency call initiated! SID:", call.sid)
        return True
    except Exception as e:
        print(f"❌ Failed to initiate emergency call: {e}")
        return False


class FallDetector:
    def __init__(self):
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75,
            model_complexity=2,
            smooth_landmarks=True,
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Fall detection state
        self.fall_detected = False
        self.emergency_active = False
        self.emergency_thread = None
        
        # Audio file paths
        pygame.mixer.init()
        self.audio_base_path = r"C:\ALL folder in dexstop\PycharmProjects\AI\fall dedection\Alert"
        self.are_you_ok_audio = os.path.join(self.audio_base_path, "Are-u-ok.mp3")
        self.emergency_audio = os.path.join(self.audio_base_path, "Emergency.mp3")

        # Speech recognition
        self.recognizer = sr.Recognizer()
        self.last_sms_time = 0
        self.sms_cooldown = 60

        # Recording feature
        self.is_recording = False
        self.record_start_time = None
        self.record_duration = 180  # 3 minutes
        self.output_video = None
        self.output_path = None
        self.recording_thread = None
        self.recording_lock = threading.Lock()
        self.stop_recording = False
        self.recording_fps = 30
        self.recording_resolution = None
        self.last_frame = None

        # Check if audio files exist
        if not os.path.exists(self.are_you_ok_audio):
            print(f"WARNING: 'Are-u-ok.mp3' not found at {self.are_you_ok_audio}")
        if not os.path.exists(self.emergency_audio):
            print(f"WARNING: 'Emergency.mp3' not found at {self.emergency_audio}")

    def reset(self):
        """Reset detection and recording state"""
        self.fall_detected = False
        self.emergency_active = False

        if self.emergency_thread and self.emergency_thread.is_alive():
            self.emergency_thread.join(timeout=1)

        # Stop recording if active
        if self.is_recording:
            self.stop_recording = True
            with self.recording_lock:
                if self.output_video:
                    self.output_video.release()
                    self.output_video = None
                self.is_recording = False
                print(f"Recording stopped and saved to {self.output_path}")

        self.last_sms_time = 0
        print("System reset complete.")

    def emergency_sequence(self):
        """Handle the emergency response sequence with voice recognition"""
        if self.emergency_active:
            return

        self.emergency_active = True
        print("Starting emergency sequence...")

        # Wait 5 seconds before playing "Are you okay?"
        time.sleep(5)
        if not self.emergency_active:  # Check if reset during wait
            return

        # Play "Are you okay?" audio
        try:
            if os.path.exists(self.are_you_ok_audio):
                pygame.mixer.music.load(self.are_you_ok_audio)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                print("Played 'Are you okay?' audio")
            else:
                import winsound

                winsound.Beep(1000, 1000)
                print("Using system beep for 'Are you okay?'")
        except Exception as e:
            print(f"Error playing 'Are you okay?' audio: {e}")

        # Listen for response for 10 seconds
        start_time = time.time()
        response_timeout = 10
        positive_responses = ["yes", "i'm fine", "okay", "ok"]
        negative_responses = ["no", "help"]

        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("Listening for response...")
                while (
                    time.time() - start_time < response_timeout
                    and self.emergency_active
                ):
                    try:
                        audio = self.recognizer.listen(
                            source, timeout=2, phrase_time_limit=3
                        )
                        response = self.recognizer.recognize_google(audio).lower()
                        print(f"Detected response: {response}")

                        if any(word in response for word in positive_responses):
                            print(
                                "Positive response detected. Stopping emergency sequence."
                            )
                            self.emergency_active = False
                            self.fall_detected = False
                            return
                        elif any(word in response for word in negative_responses):
                            print(
                                "Negative response detected. Escalating to emergency."
                            )
                            self.escalate_emergency()
                            return
                    except sr.WaitTimeoutError:
                        continue
                    except sr.UnknownValueError:
                        print("Could not understand audio")
                        continue
                    except Exception as e:
                        print(f"Speech recognition error: {e}")
                        continue
        except Exception as e:
            print(f"Microphone error: {e}")

        # No response after timeout
        if self.emergency_active:
            print("No response after 10 seconds. Escalating to emergency.")
            self.escalate_emergency()

    def escalate_emergency(self):
        """Escalate to emergency call after no positive response"""
        try:
            if os.path.exists(self.emergency_audio):
                pygame.mixer.music.load(self.emergency_audio)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                print("Played emergency audio")
            else:
                import winsound

                winsound.Beep(2000, 1000)
                print("Using system beep for emergency")
        except Exception as e:
            print(f"Error playing emergency audio: {e}")

        # Make emergency call in a separate thread
        call_thread = threading.Thread(target=make_emergency_call)
        call_thread.daemon = True
        call_thread.start()

        self.emergency_active = False

    def start_recording(self, frame):
        """Start recording video when fall is detected"""
        if self.is_recording:
            return

        output_dir = r"C:\ALL folder in dexstop\PycharmProjects\AI\fall dedection\Analysed-Clip"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = "fall_detection"
        self.output_path = os.path.join(output_dir, f"{video_name}_{timestamp}.mp4")
        height, width = frame.shape[:2]
        self.recording_resolution = (width, height)

        with self.recording_lock:
            # Try codecs in order: H.264 (avc1), H.264 alternative (H264), MPEG-4 (mp4v)
            codecs = [('avc1', 'H.264'), ('H264', 'H.264 alternative'), ('mp4v', 'MPEG-4')]
            for codec, codec_name in codecs:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    self.output_video = cv2.VideoWriter(
                        self.output_path,
                        fourcc,
                        self.recording_fps,
                        (width, height),
                        True,
                    )
                    if self.output_video.isOpened():
                        print(f"Started recording to {self.output_path} using {codec_name} codec")
                        break
                    else:
                        self.output_video = None
                        print(f"Warning: Failed to initialize VideoWriter with {codec_name} codec")
                except Exception as e:
                    print(f"Error with {codec_name} codec: {str(e)}")
                    self.output_video = None
                    continue

            # Fallback to AVI with MJPG if all codecs fail
            if not self.output_video or not self.output_video.isOpened():
                self.output_path = os.path.join(
                    output_dir, f"{video_name}_{timestamp}_fallback.avi"
                )
                try:
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    self.output_video = cv2.VideoWriter(
                        self.output_path,
                        fourcc,
                        self.recording_fps,
                        (width, height),
                        True,
                    )
                    if self.output_video.isOpened():
                        print(f"Started recording to {self.output_path} using MJPG codec (AVI)")
                    else:
                        print("ERROR: Failed to create video writer with MJPG codec")
                        self.output_video = None
                        return
                except Exception as e:
                    print(f"Error with MJPG codec: {str(e)}")
                    self.output_video = None
                    return

            self.is_recording = True
            self.record_start_time = time.time()
            self.stop_recording = False
            self.recording_thread = threading.Thread(target=self._recording_monitor)
            self.recording_thread.daemon = True
            self.recording_thread.start()

    def _recording_monitor(self):
        """Monitor recording duration and stop when exceeded"""
        while not self.stop_recording:
            if self.is_recording and self.record_start_time:
                elapsed_time = time.time() - self.record_start_time
                if elapsed_time > self.record_duration:
                    self.stop_recording = True
                    with self.recording_lock:
                        if self.output_video:
                            self.output_video.release()
                            self.output_video = None
                        self.is_recording = False
                        print(
                            f"Stopped recording after {elapsed_time:.1f} seconds. Video saved to {self.output_path}"
                        )
                    break
            time.sleep(1)

    def add_frame_to_recording(self, frame):
        """Add current frame to the recording"""
        with self.recording_lock:
            if self.is_recording and self.output_video:
                try:
                    if self.recording_resolution:
                        if (
                            frame.shape[1] != self.recording_resolution[0]
                            or frame.shape[0] != self.recording_resolution[1]
                        ):
                            frame = cv2.resize(frame, self.recording_resolution)
                    self.output_video.write(frame)
                except Exception as e:
                    print(f"Error adding frame to recording: {e}")

    def force_fall_detection(self):
        """Force a fall detection event (triggered by keyboard)"""
        self.fall_detected = True

        # Start recording
        if not self.is_recording and self.last_frame is not None:
            self.start_recording(self.last_frame)

        # Send SMS if cooldown period elapsed
        current_time = time.time()
        if current_time - self.last_sms_time >= self.sms_cooldown:
            success = send_emergency_sms()
            if success:
                self.last_sms_time = current_time

        # Start emergency sequence
        if not self.emergency_active and (
            self.emergency_thread is None or not self.emergency_thread.is_alive()
        ):
            self.emergency_thread = threading.Thread(target=self.emergency_sequence)
            self.emergency_thread.daemon = True
            self.emergency_thread.start()

    def draw_skeleton(self, frame, landmarks):
        """Draw skeleton on frame using MediaPipe landmarks"""
        h, w = frame.shape[:2]

        # Draw connections between landmarks
        self.mp_drawing.draw_landmarks(
            frame,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=2
            ),
            self.mp_drawing.DrawingSpec(
                color=(0, 0, 255), thickness=2, circle_radius=2
            ),
        )

        # Highlight key points
        if landmarks and landmarks.landmark:
            for idx, landmark in enumerate(
                [
                    self.mp_pose.PoseLandmark.NOSE,
                    self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                    self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    self.mp_pose.PoseLandmark.LEFT_HIP,
                    self.mp_pose.PoseLandmark.RIGHT_HIP,
                    self.mp_pose.PoseLandmark.LEFT_KNEE,
                    self.mp_pose.PoseLandmark.RIGHT_KNEE,
                    self.mp_pose.PoseLandmark.LEFT_ANKLE,
                    self.mp_pose.PoseLandmark.RIGHT_ANKLE,
                ]
            ):
                lm = landmarks.landmark[landmark]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    def process_frame(self, frame, frame_number):
        """Process a single frame for fall detection"""
        # Save the frame for potential recording
        self.last_frame = frame.copy()

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        # Create output frame
        output_frame = frame.copy()

        # Add timestamp at top-left (position 1)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            output_frame,
            timestamp,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # Add system mode at position 2 (below timestamp)
        cv2.putText(
            output_frame,
            "FALL DETECTION SYSTEM",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        # No keyboard shortcuts display

        if results.pose_landmarks:
            # Draw skeleton on the frame
            self.draw_skeleton(output_frame, results.pose_landmarks)

            # For testing: detect if hip is low (simplified detection)
            hip_y = results.pose_landmarks.landmark[
                self.mp_pose.PoseLandmark.LEFT_HIP
            ].y
            if hip_y > 0.7 and not self.fall_detected:
                # This is just a basic threshold for demo purposes
                # You should implement more sophisticated detection in a real system
                pass

        # Display fall status at top-center (position 4) with better visibility
        if self.fall_detected:
            # Draw white background for better visibility
            text = "FALL DETECTED!"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = output_frame.shape[1] // 2 - text_size[0] // 2
            text_y = 50
            # White outline for better visibility
            cv2.putText(
                output_frame,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                4,
            )
            # Red text
            cv2.putText(
                output_frame,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        # Display smaller recording indicator at top-right (position 5)
        if self.is_recording:
            # Small red dot
            cv2.circle(
                output_frame, (output_frame.shape[1] - 20, 20), 8, (0, 0, 255), -1
            )
            # Small REC text
            cv2.putText(
                output_frame,
                "REC",
                (output_frame.shape[1] - 50, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

            # Compact recording time
            if self.record_start_time:
                elapsed = time.time() - self.record_start_time
                rec_time = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
                cv2.putText(
                    output_frame,
                    rec_time,
                    (output_frame.shape[1] - 50, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),
                    1,
                )

        # Add frame to recording if active
        if self.is_recording:
            self.add_frame_to_recording(output_frame)

        return output_frame


def main():
    """Main function to run the fall detection system"""
    fall_detector = FallDetector()

    print("\n" + "=" * 80)
    print("Fall Detection System")
    print("=" * 80)
    print("System started. Press Q to quit.")
    print("=" * 80 + "\n")

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0
    cv2.namedWindow("Fall Detection System", cv2.WINDOW_NORMAL)

    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from camera.")
                break

            # Process frame
            processed_frame = fall_detector.process_frame(frame, frame_count)

            # Display frame
            cv2.imshow("Fall Detection System", processed_frame)
            frame_count += 1

            # Check for keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quitting...")
                break
            elif key == ord("f"):
                print("Forcing fall detection...")
                fall_detector.force_fall_detection()
            elif key == ord("a"):
                print("Testing audio playback...")
                try:
                    if os.path.exists(fall_detector.are_you_ok_audio):
                        pygame.mixer.music.load(fall_detector.are_you_ok_audio)
                        pygame.mixer.music.play()
                        print("Playing 'Are you okay?' audio")
                    else:
                        import winsound

                        winsound.Beep(1000, 1000)
                        print("Using system beep for 'Are you okay?'")
                except Exception as e:
                    print(f"Error playing audio: {e}")
            elif key == ord("c"):
                print("Testing emergency call...")
                threading.Thread(target=make_emergency_call, daemon=True).start()
            elif key == ord("s"):
                print("Testing SMS...")
                send_emergency_sms()
            elif key == ord("r"):
                print("Resetting system...")
                fall_detector.reset()

    except KeyboardInterrupt:
        print("Stopped by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        fall_detector.reset()
        pygame.mixer.quit()
        print("Fall detection system stopped.")


if __name__ == "__main__":
    main()