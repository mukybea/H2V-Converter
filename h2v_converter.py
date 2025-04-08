import os
import cv2
import numpy as np
import json
import subprocess
from scenedetect import open_video, SceneManager, ContentDetector
from scenedetect.stats_manager import StatsManager
import whisper
from ultralytics import YOLO
import mediapipe as mp
from time import time
from tqdm import tqdm
import logging

class H2VConverter:
    def __init__(self, input_video, output_dir="output", target_ratio=(9, 16),
                 model_size="base", face_detection_confidence=0.5):
        self.input_video = input_video
        self.output_dir = output_dir
        self.target_ratio = target_ratio
        self.model_size = model_size
        self.face_detection_confidence = face_detection_confidence

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        try:
            self.whisper_model = whisper.load_model(self.model_size)
        except Exception as e:
            self.logger.error(f"Error loading Whisper model: {e}")
            raise

        try:
            self.object_detector = YOLO("yolo12n.pt")
        except Exception as e:
            self.logger.error(f"Error loading YOLO model: {e}")
            raise

        self.mp_face = mp.solutions.face_detection
        self.mp_pose = mp.solutions.pose
        self.face_detector = self.mp_face.FaceDetection(
            min_detection_confidence=face_detection_confidence)
        self.pose_detector = self.mp_pose.Pose(min_detection_confidence=0.5)
        self.focus_points = []
        self.cap = cv2.VideoCapture(input_video)
        if not self.cap.isOpened():
            self.logger.error(f"Error opening video file: {input_video}")
            raise ValueError(f"Could not open video file: {input_video}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.cap.release()
        self.calculate_target_dimensions()

        os.makedirs(output_dir, exist_ok=True)

    def calculate_target_dimensions(self):
        orig_ratio = self.frame_width / self.frame_height
        target_ratio_value = self.target_ratio[0] / self.target_ratio[1]

        if orig_ratio > target_ratio_value:
            self.target_height = self.frame_height
            self.target_width = int(self.frame_height * target_ratio_value)
        else:
            self.target_width = self.frame_width
            self.target_height = int(self.frame_width / target_ratio_value)
        self.logger.debug(f"Target dimensions: {self.target_width} x {self.target_height}")

    def detect_scenes(self):
        video_manager = open_video(self.input_video)
        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager)
        scene_manager.add_detector(ContentDetector(threshold=30))
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()
        scenes = []
        for scene in scene_list:
            start_frame = int(scene[0].get_frames())
            end_frame = int(scene[1].get_frames())
            scenes.append((start_frame, end_frame))

        self.logger.info(f"Detected {len(scenes)} scenes")
        return scenes

    def extract_audio(self):
        audio_path = os.path.join(self.output_dir, "temp_audio.wav")
        command = [
            "ffmpeg", "-i", self.input_video,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000",
            "-ac", "1", audio_path, "-y"
        ]
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return audio_path
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error extracting audio: {e}")
            self.logger.error(f"FFmpeg command: {' '.join(command)}")
            raise

    def detect_speech_segments(self, audio_path):
        try:
            result = self.whisper_model.transcribe(audio_path, word_timestamps=True)
        except Exception as e:
            self.logger.error(f"Error transcribing audio with Whisper: {e}")
            raise

        speech_segments = []
        for segment in result["segments"]:
            start_time = segment["start"]
            end_time = segment["end"]
            speech_segments.append({
                "start_time": start_time,
                "end_time": end_time,
                "start_frame": int(start_time * self.fps),
                "end_frame": int(end_time * self.fps)
            })
        return speech_segments

    def is_frame_in_speech_segment(self, frame_idx, speech_segments):
        frame_time = frame_idx / self.fps
        for segment in speech_segments:
            if segment["start_time"] <= frame_time <= segment["end_time"]:
                return True
        return False

    def _get_face_detections(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = self.face_detector.process(rgb_frame)
        if face_results.detections:
            return face_results.detections
        return []

    def _get_pose_detections(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose_detector.process(rgb_frame)
        if pose_results.pose_landmarks:
            return pose_results.pose_landmarks.landmark
        return []

    def detect_active_speaker(self, frame):
        face_detections = self._get_face_detections(frame)
        pose_landmarks = self._get_pose_detections(frame)

        candidates = []

        if face_detections:
            for detection in face_detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                confidence = detection.score[0]
                candidates.append({
                    "type": "face",
                    "bbox": (x, y, width, height),
                    "confidence": confidence,
                    "center": (x + width // 2, y + height // 2)
                })

        if pose_landmarks:
            h, w, _ = frame.shape
            visible_keypoints = 0
            points = []
            important_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            for idx in important_indices:
                if pose_landmarks[idx].visibility > 0.5:
                    visible_keypoints += 1
                    points.append((int(pose_landmarks[idx].x * w), int(pose_landmarks[idx].y * h)))

            if visible_keypoints > 3 and points:
                min_x = min(p[0] for p in points)
                min_y = min(p[1] for p in points)
                max_x = max(p[0] for p in points)
                max_y = max(p[1] for p in points)
                width = max_x - min_x
                height = max_y - min_y
                min_x = max(0, min_x - width // 4)
                min_y = max(0, min_y - height // 4)
                max_x = min(w, max_x + width // 4)
                max_y = min(h, max_y + height // 4)
                candidates.append({
                    "type": "pose",
                    "bbox": (min_x, min_y, max_x - min_x, max_y - min_y),
                    "confidence": visible_keypoints / len(important_indices),
                    "center": (min_x + (max_x - min_x) // 2, min_y + (max_y - min_y) // 2)
                })

        candidates.sort(key=lambda x: x["confidence"], reverse=True)
        return candidates[0] if candidates else None

    def detect_important_objects(self, frame):
        results = self.object_detector(frame, verbose=False)
        priority_classes = {
            0: 10,
            1: 8,
            2: 8,
            15: 7,
            16: 7,
            56: 6,
            57: 6,
            58: 6,
            60: 6,
            62: 5,
            63: 5,
            64: 5,
            66: 5,
            67: 5,
        }

        candidates = []
        if results and results[0].boxes:
            for box in results[0].boxes:
                class_id = int(box.cls.item())
                confidence = box.conf.item()
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                area = w * h
                priority = priority_classes.get(class_id, 1)
                center_x = x + w // 2
                center_y = y + h // 2
                h_img, w_img, _ = frame.shape
                center_weight = 1.0 - 0.5 * (
                    abs(center_x - w_img // 2) / (w_img // 2) +
                    abs(center_y - h_img // 2) / (h_img // 2)
                ) / 2.0
                importance = priority * confidence * (area / (frame.shape[0] * frame.shape[1])) * center_weight
                candidates.append({
                    "type": "object",
                    "class_id": class_id,
                    "bbox": (x, y, w, h),
                    "center": (center_x, center_y),
                    "importance": importance
                })
        candidates.sort(key=lambda x: x["importance"], reverse=True)
        return candidates[0] if candidates else None

    def _get_focus_object(self, frame, in_speech, last_detection, last_detection_frame, tracking_active, tracker, frame_idx):
        focus_object = None

        if in_speech:
            focus_object = self.detect_active_speaker(frame)
            if focus_object:
                return focus_object, False, None  # Reset tracking

        if tracking_active:
            success, tracking_box = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in tracking_box]
                focus_object = {
                    "type": "tracked",
                    "bbox": (x, y, w, h),
                    "center": (x + w // 2, y + h // 2)
                }
                return focus_object, True, tracking_box
            else:
                return None, False, None

        if last_detection and frame_idx - last_detection_frame < 30:
            return last_detection, False, None

        focus_object = self.detect_important_objects(frame)
        if focus_object:
            x, y, w, h = focus_object["bbox"]
            tracking_box = (x, y, w, h)
            return focus_object, True, tracking_box

        return None, False, None

    def process_scene(self, scene_start, scene_end, speech_segments):
        scene_focus_points = []
        cap = cv2.VideoCapture(self.input_video)
        sample_interval = 5
        tracker = cv2.TrackerCSRT_create()
        tracking_active = False
        tracking_box = None
        last_detection = None
        last_detection_frame = -100

        for frame_idx in range(scene_start, scene_end, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                self.logger.warning(
                    f"Failed to read frame {frame_idx} in scene {scene_start}-{scene_end}.  Exiting scene processing.")
                break

            timestamp = frame_idx / self.fps
            in_speech = self.is_frame_in_speech_segment(frame_idx, speech_segments)

            if frame_idx - last_detection_frame > 90:
                tracking_active = False

            focus_object, tracking_active, tracking_box = self._get_focus_object(
                frame, in_speech, last_detection, last_detection_frame, tracking_active, tracker, frame_idx)

            if focus_object:
                center_x, center_y = focus_object["center"]
            else:
                h, w, _ = frame.shape
                center_x, center_y = w // 2, h // 2

            scene_focus_points.append({
                "frame": frame_idx,
                "timestamp": timestamp,
                "focus_x": center_x,
                "focus_y": center_y,
                "in_speech": in_speech,
                "object_type": focus_object["type"] if focus_object else "none"
            })

        cap.release()
        return scene_focus_points

    def smooth_focus_points(self, focus_points, window_size=15):
        smoothed = []
        if len(focus_points) <= window_size:
            return focus_points

        for i in range(window_size // 2):
            smoothed.append(focus_points[i])

        for i in range(window_size // 2, len(focus_points) - window_size // 2):
            window = focus_points[i - window_size // 2:i + window_size // 2 + 1]
            weights = np.array([max(1.0, window_size // 2 - abs(j - window_size // 2)) for j in range(window_size)])
            weights = weights / np.sum(weights)
            x_sum = sum(p["focus_x"] * w for p, w in zip(window, weights))
            y_sum = sum(p["focus_y"] * w for p, w in zip(window, weights))
            new_point = focus_points[i].copy()
            new_point["focus_x"] = int(x_sum)
            new_point["focus_y"] = int(y_sum)
            smoothed.append(new_point)

        for i in range(len(focus_points) - window_size // 2, len(focus_points)):
            smoothed.append(focus_points[i])
        return smoothed

    def generate_output_video(self, focus_points):
        self.logger.info("Generating output video...")
        temp_video_path = os.path.join(self.output_dir, "temp_cropped.mp4")
        cap = cv2.VideoCapture(self.input_video)
        if not cap.isOpened():
            self.logger.error(f"Error opening input video {self.input_video} for generating output.")
            raise ValueError(f"Could not open video file: {self.input_video}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            temp_video_path,
            fourcc,
            self.fps,
            (self.target_width, self.target_height)
        )
        if not out.isOpened():
            self.logger.error(f"Error creating video writer for {temp_video_path}")
            cap.release()
            raise ValueError(f"Could not create video writer for {temp_video_path}")
        frame_idx = 0
        focus_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                while focus_idx < len(focus_points) - 1 and focus_points[focus_idx + 1]["frame"] <= frame_idx:
                    focus_idx += 1

                if focus_idx < len(focus_points):
                    focus_point = focus_points[focus_idx]

                    if focus_idx < len(focus_points) - 1 and frame_idx > focus_point["frame"]:
                        next_point = focus_points[focus_idx + 1]
                        if next_point["frame"] != focus_point["frame"]:
                            t = (frame_idx - focus_point["frame"]) / (next_point["frame"] - focus_point["frame"])
                            focus_x = int(focus_point["focus_x"] * (1 - t) + next_point["focus_x"] * t)
                            focus_y = int(focus_point["focus_y"] * (1 - t) + next_point["focus_y"] * t)
                        else:
                            focus_x, focus_y = focus_point["focus_x"], focus_point["focus_y"]
                    else:
                        focus_x, focus_y = focus_point["focus_x"], focus_point["focus_y"]

                    x_start = max(0, min(focus_x - self.target_width // 2, frame.shape[1] - self.target_width))
                    y_start = max(0, min(focus_y - self.target_height // 2, frame.shape[0] - self.target_height))
                    cropped = frame[y_start:y_start + self.target_height, x_start:x_start + self.target_width]
                    out.write(cropped)
                frame_idx += 1
        except Exception as e:
            self.logger.error(f"Error processing frames: {e}")
            raise
        finally:
            cap.release()
            out.release()

        intermediate_output = os.path.join(self.output_dir, "intermediate_vertical.mp4")
        output_path = os.path.join(self.output_dir, "final_vertical.mp4")

        command1 = [
            "ffmpeg", "-i", temp_video_path,
            "-i", self.input_video,
            "-c:v", "copy",
            "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0?",
            intermediate_output, "-y"
        ]
        command2 = [
            "ffmpeg", "-i", intermediate_output,
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            "-c:a", "aac", "-b:a", "192k",
            output_path, "-y"
        ]
        try:
            subprocess.run(command1, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(command2, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running ffmpeg: {e}")
            raise

        os.remove(temp_video_path)
        os.remove(intermediate_output)
        return output_path

    def export_focus_points(self, focus_points):
        output_file = os.path.join(self.output_dir, "focus_points.json")
        try:
            with open(output_file, 'w') as f:
                json.dump(focus_points, f, indent=2)
            self.logger.info(f"Focus points exported to {output_file}")
            return output_file
        except Exception as e:
            self.logger.error(f"Error exporting focus points: {e}")
            raise

    def process(self):
        self.logger.info(f"Processing video: {self.input_video}")
        try:
            scenes = self.detect_scenes()
            audio_path = self.extract_audio()
            speech_segments = self.detect_speech_segments(audio_path)
            all_focus_points = []
            for scene_start, scene_end in tqdm(scenes, desc="Processing scenes", unit="scene"):
                scene_focus_points = self.process_scene(scene_start, scene_end, speech_segments)
                all_focus_points.extend(scene_focus_points)

            smoothed_focus_points = self.smooth_focus_points(all_focus_points)
            output_video = self.generate_output_video(smoothed_focus_points)
            focus_points_file = self.export_focus_points(smoothed_focus_points)

            self.logger.info("Processing complete!")
            self.logger.info(f"Output video: {output_video}")
            self.logger.info(f"Focus points: {focus_points_file}")
            os.remove(audio_path)
            return {
                "video": output_video,
                "focus_points": focus_points_file
            }
        except Exception as e:
            self.logger.error(f"An error occurred during processing: {e}")
            raise

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert horizontal video to vertical with smart focus tracking")
    parser.add_argument("input_video", help="Path to input horizontal video")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--ratio", default="9:16", help="Target aspect ratio (default: 9:16)")
    parser.add_argument("--model_size", default="base", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size for speech detection")
    parser.add_argument("--face_confidence", type=float, default=0.5,
                        help="Confidence threshold for face detection")

    args = parser.parse_args()
    w, h = map(int, args.ratio.split(':'))
    start_time = time()

    converter = H2VConverter(
        input_video=args.input_video,
        output_dir=args.output_dir,
        target_ratio=(w, h),
        model_size=args.model_size,
        face_detection_confidence=args.face_confidence
    )

    try:
        result = converter.process()
        end_time = time()
        print("\nConversion completed successfully!")
        print(f"Vertical video: {result['video']}")
        print(f"Focus points: {result['focus_points']}")
        print(f"Total time taken: {(end_time - start_time) / 60:.2f} minutes")
    except Exception as e:
        print(f"Unable to complete conversion: {e}")

if __name__ == "__main__":
    main()
