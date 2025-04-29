import cv2
import mediapipe as mp
import numpy as np
import os
import time
from datetime import datetime


class FallDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7, min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Fall detection parameters
        self.height_ratio_threshold = 0.45
        self.velocity_threshold = 0.07
        self.stable_frames_threshold = 15
        self.height_history = []
        self.time_history = []
        self.previous_positions = []
        self.fall_counter = 0
        self.stable_counter = 0
        self.fall_detected = False
        self.recovery_counter = 0
        self.context_history = []
        self.max_context_history = 45
        self.falls_detected = []
        self.last_detected_state = "unknown"

    def reset(self):
        self.height_history = []
        self.time_history = []
        self.previous_positions = []
        self.fall_counter = 0
        self.stable_counter = 0
        self.fall_detected = False
        self.recovery_counter = 0
        self.context_history = []
        self.falls_detected = []
        self.last_detected_state = "unknown"

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def calculate_distance(self, point1, point2):
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def get_body_keypoints(self, landmarks):
        keypoints = {}
        keypoints["nose"] = [landmarks[self.mp_pose.PoseLandmark.NOSE].x, landmarks[self.mp_pose.PoseLandmark.NOSE].y]
        keypoints["left_eye"] = [landmarks[self.mp_pose.PoseLandmark.LEFT_EYE].x, landmarks[self.mp_pose.PoseLandmark.LEFT_EYE].y]
        keypoints["right_eye"] = [landmarks[self.mp_pose.PoseLandmark.RIGHT_EYE].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_EYE].y]
        keypoints["left_shoulder"] = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        keypoints["right_shoulder"] = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
        keypoints["left_hip"] = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y]
        keypoints["right_hip"] = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y]
        keypoints["left_knee"] = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].y]
        keypoints["right_knee"] = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE].y]
        keypoints["left_ankle"] = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].y]
        keypoints["right_ankle"] = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].y]
        keypoints["left_wrist"] = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].y]
        keypoints["right_wrist"] = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].y]
        keypoints["mid_shoulder"] = [
            (keypoints["left_shoulder"][0] + keypoints["right_shoulder"][0]) / 2,
            (keypoints["left_shoulder"][1] + keypoints["right_shoulder"][1]) / 2,
        ]
        keypoints["mid_hip"] = [
            (keypoints["left_hip"][0] + keypoints["right_hip"][0]) / 2,
            (keypoints["left_hip"][1] + keypoints["right_hip"][1]) / 2,
        ]
        return keypoints

    def detect_abnormal_posture(self, keypoints):
        spine_angle = self.calculate_angle(
            keypoints["mid_shoulder"], keypoints["mid_hip"], [keypoints["mid_hip"][0], keypoints["mid_hip"][1] + 0.1]
        )
        left_leg_angle = self.calculate_angle(keypoints["left_hip"], keypoints["left_knee"], keypoints["left_ankle"])
        right_leg_angle = self.calculate_angle(keypoints["right_hip"], keypoints["right_knee"], keypoints["right_ankle"])
        shoulder_angle = self.calculate_angle(
            [keypoints["left_shoulder"][0], keypoints["left_shoulder"][1] - 0.1], keypoints["left_shoulder"], keypoints["right_shoulder"]
        )
        is_horizontal = abs(90 - spine_angle) < 25
        legs_bent = left_leg_angle < 70 and right_leg_angle < 70
        arms_extended = self.calculate_distance(keypoints["left_shoulder"], keypoints["left_wrist"]) > 1.8 * self.calculate_distance(
            keypoints["left_shoulder"], keypoints["left_hip"]
        ) or self.calculate_distance(keypoints["right_shoulder"], keypoints["right_wrist"]) > 1.8 * self.calculate_distance(
            keypoints["right_shoulder"], keypoints["right_hip"]
        )
        is_floor_sitting = self._detect_floor_sitting(keypoints)
        return (is_horizontal or (legs_bent and arms_extended)) and not is_floor_sitting

    def detect_lying_position(self, keypoints):
        head_y = min(keypoints["nose"][1], keypoints["left_eye"][1], keypoints["right_eye"][1])
        feet_y = max(keypoints["left_ankle"][1], keypoints["right_ankle"][1])
        body_height_ratio = feet_y - head_y
        self.height_history.append(body_height_ratio)
        if len(self.height_history) > 30:
            self.height_history.pop(0)
        shoulder_hip_dist = self.calculate_distance(keypoints["mid_shoulder"], keypoints["mid_hip"])
        hip_ankle_dist = (
            self.calculate_distance(keypoints["left_hip"], keypoints["left_ankle"]) + self.calculate_distance(keypoints["right_hip"], keypoints["right_ankle"])
        ) / 2
        ankles_near_hips = (
            self.calculate_distance(keypoints["left_hip"], keypoints["left_ankle"]) < 0.4 * hip_ankle_dist
            or self.calculate_distance(keypoints["right_hip"], keypoints["right_ankle"]) < 0.4 * hip_ankle_dist
        )
        is_sitting_on_floor = (
            keypoints["mid_hip"][1] > keypoints["mid_shoulder"][1]
            and keypoints["left_knee"][1] > keypoints["left_hip"][1]
            and keypoints["right_knee"][1] > keypoints["right_hip"][1]
        )
        is_lying = body_height_ratio < self.height_ratio_threshold
        return is_lying and not (ankles_near_hips and is_sitting_on_floor)

    def detect_sudden_movement(self, keypoints):
        current_time = time.time()
        self.time_history.append(current_time)
        if len(self.time_history) > 10:
            self.time_history.pop(0)
        if len(self.time_history) < 2:
            return False
        time_diff = self.time_history[-1] - self.time_history[0]
        if time_diff == 0:
            return False
        mid_hip = keypoints["mid_hip"]
        self.previous_positions.append((mid_hip[0], mid_hip[1]))
        if len(self.previous_positions) > 15:
            self.previous_positions.pop(0)
        if len(self.height_history) >= 2:
            height_diff = abs(self.height_history[-1] - self.height_history[0])
            velocity = height_diff / time_diff
            if velocity > self.velocity_threshold:
                is_controlled = self._is_controlled_movement(keypoints)
                return not is_controlled and velocity > self.velocity_threshold * 1.5
        return False

    def _is_controlled_movement(self, keypoints):
        if len(self.previous_positions) < 8:
            return False
        trajectory_smoothness = 0
        for i in range(1, len(self.previous_positions) - 1):
            prev = np.array(self.previous_positions[i - 1])
            current = np.array(self.previous_positions[i])
            next_pos = np.array(self.previous_positions[i + 1])
            v1 = current - prev
            v2 = next_pos - current
            if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
                continue
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle_change = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            trajectory_smoothness += angle_change
        avg_smoothness = trajectory_smoothness / (len(self.previous_positions) - 2) if len(self.previous_positions) > 2 else 0
        return avg_smoothness < 0.3

    def measure_stability(self, keypoints):
        com_x = (keypoints["mid_shoulder"][0] + keypoints["mid_hip"][0]) / 2
        com_y = (keypoints["mid_shoulder"][1] + keypoints["mid_hip"][1]) / 2
        left_ankle = keypoints["left_ankle"]
        right_ankle = keypoints["right_ankle"]
        left_knee = keypoints["left_knee"]
        right_knee = keypoints["right_knee"]
        hip_height = keypoints["mid_hip"][1]
        knee_height = (keypoints["left_knee"][1] + keypoints["right_knee"][1]) / 2
        ankle_height = (keypoints["left_ankle"][1] + keypoints["right_ankle"][1]) / 2
        sitting_indicator = hip_height > 0.7 * ankle_height
        knees_wide = self.calculate_distance(keypoints["left_knee"], keypoints["right_knee"]) > 0.3
        if sitting_indicator and knees_wide:
            support_points = [left_knee, right_knee, left_ankle, right_ankle]
            min_x = min(p[0] for p in support_points)
            max_x = max(p[0] for p in support_points)
            min_x -= 0.05
            max_x += 0.05
            is_unstable = com_x < min_x or com_x > max_x
            return is_unstable and not sitting_indicator
        else:
            min_ankle_x = min(left_ankle[0], right_ankle[0])
            max_ankle_x = max(left_ankle[0], right_ankle[0])
            is_unstable = com_x < min_ankle_x - 0.05 or com_x > max_ankle_x + 0.05
            return is_unstable

    def _detect_floor_sitting(self, keypoints):
        cross_legged = (
            self.calculate_distance(keypoints["left_knee"], keypoints["right_knee"]) > 1.2 * self.calculate_distance(keypoints["left_hip"], keypoints["right_hip"])
            and self.calculate_distance(keypoints["left_ankle"], keypoints["right_ankle"]) < 0.7 * self.calculate_distance(keypoints["left_hip"], keypoints["right_hip"])
            and keypoints["left_knee"][1] > keypoints["left_hip"][1]
            and keypoints["right_knee"][1] > keypoints["right_hip"][1]
        )
        kneeling = (
            self.calculate_distance(keypoints["left_ankle"], keypoints["left_hip"]) < 0.5 * self.calculate_distance(keypoints["left_hip"], keypoints["left_shoulder"])
            and self.calculate_distance(keypoints["right_ankle"], keypoints["right_hip"]) < 0.5 * self.calculate_distance(keypoints["right_hip"], keypoints["right_shoulder"])
            and keypoints["left_knee"][1] > 0.85
            and keypoints["right_knee"][1] > 0.85
        )
        squatting = (
            keypoints["left_knee"][1] < keypoints["left_ankle"][1]
            and keypoints["right_knee"][1] < keypoints["right_ankle"][1]
            and self.calculate_angle(keypoints["left_hip"], keypoints["left_knee"], keypoints["left_ankle"]) < 80
            and self.calculate_angle(keypoints["right_hip"], keypoints["right_knee"], keypoints["right_ankle"]) < 80
            and keypoints["mid_hip"][1] > 0.7
        )
        sitting_on_floor = (
            keypoints["mid_hip"][1] > 0.65
            and abs(self.calculate_angle(keypoints["mid_shoulder"], keypoints["mid_hip"], [keypoints["mid_hip"][0], keypoints["mid_hip"][1] + 0.1]) - 180) < 30
        )
        return cross_legged or kneeling or squatting or sitting_on_floor

    def _is_transitioning(self, from_state, to_state):
        if len(self.context_history) < 10:
            return False
        earlier_frames = self.context_history[:-5]
        from_count = earlier_frames.count(from_state)
        recent_frames = self.context_history[-5:]
        to_count = recent_frames.count(to_state)
        return from_count > len(earlier_frames) * 0.6 and to_count > len(recent_frames) * 0.6

    def _was_recently(self, state):
        if len(self.context_history) < 15:
            return False
        recent_frames = self.context_history[-15:]
        state_count = recent_frames.count(state)
        return state_count > 5

    def process_frame(self, frame, frame_number, show_display=True):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        abnormal_posture = False
        sudden_movement = False
        lying_position = False
        unstable = False
        if result.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                result.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
            )
            landmarks = result.pose_landmarks.landmark
            keypoints = self.get_body_keypoints(landmarks)
            is_floor_sitting = self._detect_floor_sitting(keypoints)
            if is_floor_sitting:
                current_context = "floor_sitting"
            else:
                current_context = "standing"
            self.context_history.append(current_context)
            if len(self.context_history) > self.max_context_history:
                self.context_history.pop(0)
            abnormal_posture = self.detect_abnormal_posture(keypoints)
            lying_position = self.detect_lying_position(keypoints)
            sudden_movement = self.detect_sudden_movement(keypoints)
            unstable = self.measure_stability(keypoints)
            transitioning_up = self._is_transitioning("floor_sitting", "standing") and current_context == "standing"
            if transitioning_up:
                detection_factors = sum([
                    abnormal_posture and sudden_movement,
                    lying_position and unstable,
                    sudden_movement and unstable and not self._is_controlled_movement(keypoints),
                ])
            else:
                detection_factors = sum([abnormal_posture, lying_position, sudden_movement, unstable])
                if self._was_recently("floor_sitting") and detection_factors < 3:
                    detection_factors = 0
            threshold = 3 if transitioning_up else 2
            if detection_factors >= threshold:
                self.fall_counter += 1
                self.stable_counter = 0
            else:
                self.fall_counter = max(0, self.fall_counter - 1)
                self.stable_counter += 1
            required_fall_frames = 7 if transitioning_up else 5
            if self.fall_counter >= required_fall_frames and not self.fall_detected:
                self.fall_detected = True
                self.falls_detected.append(frame_number)
            if self.stable_counter >= self.stable_frames_threshold and self.fall_detected:
                self.fall_detected = False
            self.last_detected_state = current_context
            if show_display:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    image, f"Abnormal Posture: {abnormal_posture}", (10, 30), font, 0.7, (0, 0, 255) if abnormal_posture else (0, 255, 0), 2
                )
                cv2.putText(
                    image, f"Lying Position: {lying_position}", (10, 60), font, 0.7, (0, 0, 255) if lying_position else (0, 255, 0), 2
                )
                cv2.putText(
                    image, f"Sudden Movement: {sudden_movement}", (10, 90), font, 0.7, (0, 0, 255) if sudden_movement else (0, 255, 0), 2
                )
                cv2.putText(
                    image, f"Unstable: {unstable}", (10, 120), font, 0.7, (0, 0, 255) if unstable else (0, 255, 0), 2
                )
                cv2.putText(
                    image, f"Current activity: {current_context.replace('_', ' ')}", (10, 180), font, 0.7, (0, 150, 255), 2
                )
                if self.fall_detected:
                    cv2.putText(image, "FALL DETECTED!", (50, 230), font, 1.5, (0, 0, 255), 3)
                cv2.putText(image, f"Frame: {frame_number}", (10, 270), font, 0.7, (255, 255, 255), 2)
        else:
            if show_display:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image, "No person detected", (50, 50), font, 1, (0, 0, 255), 2)
                cv2.putText(image, f"Frame: {frame_number}", (10, 270), font, 0.7, (255, 255, 255), 2)
        return image, self.fall_detected

    def process_video(
        self,
        video_path,
        output_path=None,
        show_preview=True,
        start_percent=0,
        end_percent=100,
        display_width=None,
        display_height=None,
    ):
        """
        Process a video file for fall detection with options to:
        - Process only a portion of the video (start_percent to end_percent)
        - Resize the display window (display_width, display_height)
        """
        self.reset()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(
            f"Video dimensions: {width}x{height}, FPS: {fps}, Total frames: {total_frames}"
        )

        # Calculate start and end frames based on percentages
        start_frame = int((start_percent / 100) * total_frames)
        end_frame = int((end_percent / 100) * total_frames)

        print(
            f"Processing frames from {start_frame} to {end_frame} ({start_percent}% to {end_percent}%)"
        )

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Calculate display dimensions if specified
        display_size = None
        if display_width and display_height:
            display_size = (display_width, display_height)
        elif display_width:
            display_height = int(height * (display_width / width))
            display_size = (display_width, display_height)
        elif display_height:
            display_width = int(width * (display_height / height))
            display_size = (display_width, display_height)

        # Create window name and configure window properties
        window_name = f"Processing {os.path.basename(video_path)}"
        if show_preview:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            if display_size:
                cv2.resizeWindow(window_name, display_size[0], display_size[1])

        # Create output video writer if needed
        out = None
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            # Ensure .mp4 extension for web compatibility
            if not output_path.lower().endswith('.mp4'):
                output_path = os.path.splitext(output_path)[0] + '.mp4'

            # Try H.264 codec (avc1) first for web compatibility
            codecs = [('avc1', 'H.264'), ('H264', 'H.264 alternative'), ('mp4v', 'MPEG-4')]
            for codec, codec_name in codecs:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    if out.isOpened():
                        print(f"Saving processed video to: {output_path} using {codec_name} codec")
                        break
                    else:
                        out = None
                        print(f"Warning: Failed to initialize VideoWriter with {codec_name} codec")
                except Exception as e:
                    print(f"Error with {codec_name} codec: {str(e)}")
                    out = None
                    continue

            if out is None:
                print("Error: Could not initialize VideoWriter with any codec. No output will be saved.")

        # Process each frame
        frame_number = start_frame
        fall_detected = False

        while cap.isOpened() and frame_number < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame
            processed_frame, current_fall_detected = self.process_frame(
                frame, frame_number, show_preview
            )

            # Update overall fall detection status
            if current_fall_detected:
                fall_detected = True

            # Write to output video
            if out and processed_frame is not None:
                out.write(processed_frame)

            # Show preview if requested
            if show_preview:
                if display_size:
                    display_frame = cv2.resize(processed_frame, display_size)
                    cv2.imshow(window_name, display_frame)
                else:
                    cv2.imshow(window_name, processed_frame)

                # Display progress
                total_to_process = end_frame - start_frame
                progress = (
                    ((frame_number - start_frame) / total_to_process) * 100
                    if total_to_process > 0
                    else 0
                )
                print(
                    f"\rProcessing {os.path.basename(video_path)}: {progress:.1f}% ({frame_number}/{end_frame})",
                    end="",
                )

                # Break on 'q' press
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_number += 1

        # Clean up
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        print("")  # New line after progress display

        # Return analysis results
        result = {
            "video_name": os.path.basename(video_path),
            "fall_detected": fall_detected,
            "fall_frames": self.falls_detected,
            "total_frames_processed": frame_number - start_frame,
            "total_frames": total_frames,
            "processed_portion": f"{start_percent}% to {end_percent}%",
        }

        return result


def batch_process_videos(
    video_dir,
    output_dir=None,
    start_num=1,
    end_num=31,
    start_percent=0,
    end_percent=100,
    display_width=None,
    display_height=None,
):
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    detector = FallDetector()
    results = []
    extensions = [".mov", ".mp4", ".avi", ".mkv"]

    for i in range(start_num, end_num + 1):
        video_path = None
        for ext in extensions:
            test_path = os.path.join(video_dir, f"{i}{ext}")
            if os.path.exists(test_path):
                video_path = test_path
                break

        if not video_path:
            print(f"Warning: Video {i} not found with any supported extension, skipping...")
            continue

        print(f"\n{'='*50}")
        print(f"Processing video {i} of {end_num}: {os.path.basename(video_path)}")
        print(f"Processing portion: {start_percent}% to {end_percent}%")
        if display_width or display_height:
            print(f"Display size: {display_width}x{display_height}")
        print(f"{'='*50}")

        output_path = None
        if output_dir:
            output_path = os.path.join(output_dir, f"{i}_analyzed.mp4")

        result = detector.process_video(
            video_path,
            output_path,
            start_percent=start_percent,
            end_percent=end_percent,
            display_width=display_width,
            display_height=display_height,
        )

        if result:
            results.append(result)
            if result["fall_detected"]:
                print(f"✓ Fall detected in video {i} at frames: {result['fall_frames']}")
            else:
                print(f"✗ No falls detected in video {i}")

    report_path = os.path.join(output_dir if output_dir else video_dir, "fall_detection_report.txt")
    with open(report_path, "w") as f:
        f.write("Fall Detection Analysis Report\n")
        f.write("=============================\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Video portion analyzed: {start_percent}% to {end_percent}%\n\n")
        total_videos = len(results)
        videos_with_falls = sum(1 for r in results if r["fall_detected"])
        f.write(f"Total videos analyzed: {total_videos}\n")
        f.write(f"Videos with falls detected: {videos_with_falls}\n")
        f.write(f"Detection rate: {videos_with_falls/total_videos*100:.1f}%\n\n")
        f.write("Detailed Results:\n")
        f.write("----------------\n")
        for r in results:
            f.write(f"Video: {r['video_name']}\n")
            f.write(f"Fall detected: {'Yes' if r['fall_detected'] else 'No'}\n")
            f.write(f"Processed {r['total_frames_processed']} of {r['total_frames']} frames ({r['processed_portion']})\n")
            if r["fall_detected"]:
                f.write(f"Fall frames: {r['fall_frames']}\n")
                for frame in r["fall_frames"]:
                    time_in_video = frame / 30
                    f.write(f"  - Frame {frame} (approx. {time_in_video:.2f} seconds into video)\n")
            f.write("\n")

    print(f"\nAnalysis complete! Report saved to {report_path}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch Fall Detection for Multiple Videos")
    parser.add_argument(
        "--video_dir",
        type=str,
        default=r"C:\Users\siddh\Downloads\fall clip",
        help="Directory containing the video files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"C:\Users\siddh\Downloads\fall Ai",
        help="Directory to save analyzed videos",
    )
    parser.add_argument("--start", type=int, default=1, help="Starting video number")
    parser.add_argument("--end", type=int, default=31, help="Ending video number")
    parser.add_argument("--no_preview", action="store_true", help="Disable video preview window")
    parser.add_argument(
        "--start_percent",
        type=float,
        default=0,
        help="Start processing from this percentage of the video (0-100)",
    )
    parser.add_argument(
        "--end_percent",
        type=float,
        default=100,
        help="End processing at this percentage of the video (0-100)",
    )
    parser.add_argument(
        "--display_width",
        type=int,
        default=None,
        help="Width of display window (maintains aspect ratio if only one dimension provided)",
    )
    parser.add_argument(
        "--display_height",
        type=int,
        default=None,
        help="Height of display window (maintains aspect ratio if only one dimension provided)",
    )

    args = parser.parse_args()

    print(f"Processing videos from: {args.video_dir}")
    print(f"Saving results to: {args.output_dir}")
    print(f"Video range: {args.start} to {args.end}")
    print(f"Processing video portions: {args.start_percent}% to {args.end_percent}%")
    if args.display_width or args.display_height:
        print(f"Display size: {args.display_width}x{args.display_height}")

    batch_process_videos(
        args.video_dir,
        args.output_dir,
        args.start,
        args.end,
        args.start_percent,
        args.end_percent,
        args.display_width,
        args.display_height,
    )