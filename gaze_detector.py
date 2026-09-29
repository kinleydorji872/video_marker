"""Local 3D gaze features using MediaPipe Face Landmarker.

The detector is deliberately evidence-oriented: it reports gaze geometry and a
probabilistic reading proxy, never an assertion about presenter intent.
"""

from pathlib import Path
import os

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:  # Optional until the local MediaPipe wheel is installed.
    mp = None


class MediaPipeGazeDetector:
    LEFT_IRIS = (468, 469, 470, 471, 472)
    RIGHT_IRIS = (473, 474, 475, 476, 477)

    def __init__(self, model_path=None, sample_interval_sec=0.5, sample_rate=None):
        self.sample_interval_sec = max(0.1, float(sample_interval_sec))
        self.sample_rate = sample_rate
        configured_path = model_path or os.getenv("MEDIAPIPE_FACE_MODEL", "models/face_landmarker.task")
        self.model_path = Path(configured_path)
        self.landmarker = None
        if mp is not None and self.model_path.exists():
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_faces=1,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=True,
            )
            self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    @property
    def available(self):
        return self.landmarker is not None

    @staticmethod
    def _point(landmarks, index):
        point = landmarks[index]
        return np.array([point.x, point.y, point.z], dtype=np.float32)

    def _eye_measurement(self, landmarks, iris_indices, corner_indices):
        iris = np.mean([self._point(landmarks, index) for index in iris_indices], axis=0)
        left_corner = self._point(landmarks, corner_indices[0])
        right_corner = self._point(landmarks, corner_indices[1])
        eye_width = max(float(np.linalg.norm(right_corner - left_corner)), 1e-6)
        horizontal_ratio = float(np.dot(iris - left_corner, right_corner - left_corner) / (eye_width * eye_width))
        eye_center_y = (left_corner[1] + right_corner[1]) / 2.0
        upper = self._point(landmarks, corner_indices[2])
        lower = self._point(landmarks, corner_indices[3])
        eye_height = max(abs(float(lower[1] - upper[1])), eye_width * 0.15, 1e-6)
        downward_ratio = max(0.0, float((iris[1] - eye_center_y) / eye_height))
        return horizontal_ratio, downward_ratio

    def analyze_video(self, video_path):
        if not self.available:
            return self.empty_result("MediaPipe Face Landmarker model is unavailable")

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            return self.empty_result("Video could not be opened")
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        horizontal = []
        downward = []
        sampled_frames = 0
        failed_frames = 0
        frame_index = 0
        frame_step = max(1, int(round(fps * self.sample_interval_sec)))
        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                if frame_index % frame_step:
                    frame_index += 1
                    continue
                sampled_frames += 1
                timestamp_ms = int(frame_index * 1000 / fps)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = self.landmarker.detect_for_video(image, timestamp_ms)
                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]
                    try:
                        left = self._eye_measurement(landmarks, self.LEFT_IRIS, (33, 133, 159, 145))
                        right = self._eye_measurement(landmarks, self.RIGHT_IRIS, (362, 263, 386, 374))
                        horizontal.append((left[0] + right[0]) / 2.0)
                        downward.append((left[1] + right[1]) / 2.0)
                    except (IndexError, ValueError):
                        failed_frames += 1
                else:
                    failed_frames += 1
                frame_index += 1
        finally:
            capture.release()

        if len(horizontal) < 3:
            return self.empty_result("Insufficient face landmark evidence", len(horizontal), sampled_frames, failed_frames)
        gaze_variance = float(np.var(horizontal))
        average_downward = float(np.mean(downward))
        downward_ratio = float(np.mean(np.asarray(downward) > 0.35))
        reading_score = np.clip(gaze_variance * 1800 + average_downward * 75 + downward_ratio * 35, 0, 100)
        return {
            "available": True,
            "samples": len(horizontal),
            "sampled_frames": sampled_frames,
            "failed_frames": failed_frames,
            "low_confidence": failed_frames / max(sampled_frames, 1) > 0.30,
            "gaze_variance": round(gaze_variance, 6),
            "horizontal_gaze_ratio": round(float(np.mean(horizontal)), 4),
            "downward_iris_displacement": round(average_downward, 4),
            "downward_gaze_ratio": round(downward_ratio, 4),
            "reading_confidence_score": round(float(reading_score), 2),
            "note": "Gaze is a probabilistic visual proxy, not proof of reading or presenter intent.",
        }

    @staticmethod
    def empty_result(note, samples=0, sampled_frames=0, failed_frames=0):
        return {
            "available": False,
            "samples": samples,
            "sampled_frames": sampled_frames,
            "failed_frames": failed_frames,
            "low_confidence": True,
            "gaze_variance": 0.0,
            "horizontal_gaze_ratio": 0.0,
            "downward_iris_displacement": 0.0,
            "downward_gaze_ratio": 0.0,
            "reading_confidence_score": 0.0,
            "note": note,
        }


GazeDetector = MediaPipeGazeDetector
