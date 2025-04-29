import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import time
from datetime import datetime
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from gtts import gTTS
import speech_recognition as sr
import os
import pygame
from twilio.rest import Client
import mediapipe as mp


class Settings:
    YOLO_MODEL = "yolov8s.pt"
    CONFIDENCE_THRESHOLD = 0.6
    CAMERA_SOURCE = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    MIN_BOX_WIDTH = 50
    MIN_BOX_HEIGHT = 50
    MAX_BOX_WIDTH = int(FRAME_WIDTH * 0.9)
    MAX_BOX_HEIGHT = int(FRAME_HEIGHT * 0.9)
    FALL_CONFIRMATION_TIME = 3

    SENDER_EMAIL = "siddharth.r.college@gmail.com"
    RECEIVER_EMAIL = "cloud001abc@gmail.com"
    EMAIL_PASSWORD = "sIdDharth@[54321]"

    TWILIO_ACCOUNT_SID = "your_twilio_account_sid"
    TWILIO_AUTH_TOKEN = "your_twilio_auth_token"
    TWILIO_PHONE_NUMBER = "your_twilio_phone_number"

    EMERGENCY_CONTACTS = ["+1234567890", "+1234567891"]
    RESPONSE_TIMEOUT = 60
    POSITIVE_RESPONSES = [
        "ok",
        "i am ok",
        "i'm ok",
        "good",
        "i am good",
        "i'm good",
        "fine",
        "get up",
    ]
    RECORDING_PATH = "C:\\Users\\siddh\\Downloads\\fall"


class FallDetectionSystem:
    def speak_alert(self, text):
        try:
            tts = gTTS(text=text, lang="en")
            tts.save("alert.mp3")
            pygame.mixer.music.load("alert.mp3")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(1)
            os.remove("alert.mp3")
        except Exception as e:
            print(f"Failed to play voice alert: {e}")

    def listen_for_response(self):
        print("Listening for response...")
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source, timeout=10)
                response = self.recognizer.recognize_google(audio).lower()
                print(f"Heard response: {response}")
                return any(phrase in response for phrase in Settings.POSITIVE_RESPONSES)
        except Exception as e:
            print(f"No clear response heard: {e}")
            return False

    def __init__(self):
        print("Initializing Fall Detection System...")
        self.model = YOLO(Settings.YOLO_MODEL)
        self.cap = cv2.VideoCapture(Settings.CAMERA_SOURCE)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Settings.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Settings.FRAME_HEIGHT)

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.fall_detected = False
        self.fall_start_time = None
        self.last_positions = []

        pygame.mixer.init()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.twilio_client = Client(
            Settings.TWILIO_ACCOUNT_SID, Settings.TWILIO_AUTH_TOKEN
        )

        if not os.path.exists(Settings.RECORDING_PATH):
            os.makedirs(Settings.RECORDING_PATH)

    def send_sms_alert(self):
        try:
            message_body = f"Fall detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Please check immediately."
            for phone_number in Settings.EMERGENCY_CONTACTS:
                self.twilio_client.messages.create(
                    body=message_body,
                    from_=Settings.TWILIO_PHONE_NUMBER,
                    to=phone_number,
                )
            print("SMS alert sent successfully")
        except Exception as e:
            print(f"Failed to send SMS alert: {e}")

    def handle_fall(self):
        self.send_sms_alert()
        self.speak_alert("Fall detected! Are you okay? Please respond if you're okay.")
        self.send_sms_alert()
        self.speak_alert("Fall detected! Are you okay? Please respond if you're okay.")

        response_received = False
        start_time = time.time()

        while time.time() - start_time < Settings.RESPONSE_TIMEOUT:
            if self.listen_for_response():
                response_received = True
                self.speak_alert("Thank you for responding. Stay safe!")
                return
            time.sleep(1)

        # If no response for more than 1 minute, start SOS call
        if not response_received and time.time() - start_time > 60:
            self.speak_alert(
                "No response received. Sending SOS and contacting emergency services."
            )
            threading.Thread(target=self.make_emergency_calls).start()
            if self.listen_for_response():
                response_received = True
                self.speak_alert("Thank you for responding. Stay safe!")
                return
            time.sleep(1)

        if not response_received:
            self.speak_alert(
                "No response received. Sending SOS and contacting emergency services."
            )
            threading.Thread(target=self.make_emergency_calls).start()
            while True:
                self.speak_alert(
                    "Emergency! Person has fallen and is not responding. Please help!"
                )
                time.sleep(30)

    def detect_fall(self, frame):
        results = self.model(frame)
        if not results or len(results[0].boxes) == 0:
            return False, frame

        for result in results[0].boxes.data:
            x1, y1, x2, y2, conf, _ = result.tolist()
            if conf < Settings.CONFIDENCE_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            width, height = x2 - x1, y2 - y1

            if width > height:
                self.last_positions.append((x1, y1))
                if len(self.last_positions) > 10:
                    self.last_positions.pop(0)

                if len(self.last_positions) > 5:
                    movement = np.linalg.norm(
                        np.array(self.last_positions[-1])
                        - np.array(self.last_positions[0])
                    )
                    if movement < 15:
                        if not self.fall_detected:
                            self.fall_detected = True
                            self.fall_start_time = time.time()
                        elif (
                            time.time() - self.fall_start_time
                            > Settings.FALL_CONFIRMATION_TIME
                        ):
                            self.handle_fall()
                            return True, frame

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255) if self.fall_detected else (0, 255, 0),
                2,
            )
            cvzone.putTextRect(
                frame,
                "FALL DETECTED!" if self.fall_detected else "Normal",
                (x1, y1 - 10),
                1,
                1,
            )

        return False, frame

    def run(self):
        print("Starting fall detection. Press 'q' to quit.")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            fall_detected, frame = self.detect_fall(frame)
            if fall_detected:
                cvzone.putTextRect(
                    frame, "FALL DETECTED!", (50, 50), 2, 2, colorR=(255, 0, 0)
                )

            cv2.imshow("Fall Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    system = FallDetectionSystem()
    system.run()
