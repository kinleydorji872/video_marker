import cv2
import numpy as np
from pathlib import Path

class VideoFeatureExtractor:
    def __init__(self, sample_rate=60, gaze_detector=None):
        self.sample_rate = sample_rate
        self.gaze_detector = gaze_detector
        cascade_root = cv2.data.haarcascades
        self.face_detector = None
        self.eye_detector = None
        if hasattr(cv2, "CascadeClassifier"):
            face_path = Path(cascade_root) / "haarcascade_frontalface_default.xml"
            eye_path = Path(cascade_root) / "haarcascade_eye_tree_eyeglasses.xml"
            if face_path.exists():
                self.face_detector = cv2.CascadeClassifier(str(face_path))
            if eye_path.exists():
                self.eye_detector = cv2.CascadeClassifier(str(eye_path))

    def extract_features(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        sharpness_list = []
        brightness_list = []
        motion_list = []
        face_samples = 0
        eye_positions = []
        downward_gaze_samples = 0

        prev_gray = None
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.sample_rate == 0:
                # Convert to Grayscale for low CPU consumption
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # 1. Measure Sharpness (Laplacian Variance)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                sharpness_list.append(sharpness)

                # 2. Measure Brightness (Mean Pixel Intensity)
                brightness = np.mean(gray)
                brightness_list.append(brightness)

                # Eye-line behavior is a proxy only; it cannot prove that a presenter is reading.
                faces = []
                if self.face_detector is not None and not self.face_detector.empty():
                    faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
                if len(faces):
                    face_x, face_y, face_width, face_height = max(faces, key=lambda item: item[2] * item[3])
                    face_samples += 1
                    face_region = gray[face_y:face_y + int(face_height * 0.62), face_x:face_x + face_width]
                    eyes = []
                    if self.eye_detector is not None and not self.eye_detector.empty():
                        eyes = self.eye_detector.detectMultiScale(face_region, scaleFactor=1.1, minNeighbors=4, minSize=(12, 12))
                    if len(eyes):
                        eye_x, eye_y, eye_width, eye_height = max(eyes, key=lambda item: item[2] * item[3])
                        normalized_x = (eye_x + eye_width / 2) / max(face_width, 1)
                        normalized_y = (eye_y + eye_height / 2) / max(face_height * 0.62, 1)
                        eye_positions.append((normalized_x, normalized_y))
                        if normalized_y > 0.42:
                            downward_gaze_samples += 1

                # 3. Measure Motion/Stability (Optical Flow Magnitude)
                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None, 
                        0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    motion_list.append(np.mean(magnitude))
                
                prev_gray = gray

            frame_idx += 1

        cap.release()

        gaze_metrics = self.gaze_detector.analyze_video(video_path) if self.gaze_detector is not None else {}

        # Aggregate averages safely
        avg_sharpness = float(np.mean(sharpness_list)) if sharpness_list else 0.0
        avg_brightness = float(np.mean(brightness_list)) if brightness_list else 0.0
        avg_motion = float(np.mean(motion_list)) if motion_list else 0.0
        gaze_shift = float(np.std([position[0] for position in eye_positions])) if len(eye_positions) > 1 else 0.0
        eye_samples = len(eye_positions)
        downward_ratio = downward_gaze_samples / max(eye_samples, 1)
        face_visibility = face_samples / max(len(sharpness_list), 1)
        reading_likelihood = min(100, max(0, gaze_shift * 420 + downward_ratio * 35)) if eye_samples >= 3 else 0
        if face_visibility < 0.2 or eye_samples < 3:
            reading_status = "Insufficient camera evidence"
        elif reading_likelihood >= 62:
            reading_status = "Likely reading or checking notes"
        elif reading_likelihood >= 35:
            reading_status = "Mixed eye-line behavior"
        else:
            reading_status = "Likely direct delivery"

        return {
            "duration_sec": round(duration, 2),
            "resolution": f"{width}x{height}",
            "avg_sharpness": round(avg_sharpness, 2),
            "avg_brightness": round(avg_brightness, 2),
            "avg_motion": round(avg_motion, 2),
            "reading_behavior": {
                "status": reading_status,
                "likelihood": round(reading_likelihood, 1),
                "confidence": round(min(100, face_visibility * 100), 1),
                "face_visibility": round(face_visibility, 3),
                "eye_line_shift": round(gaze_shift, 3),
                "downward_gaze_ratio": round(downward_ratio, 3),
                "note": "This is a visual proxy, not proof of script use or presenter intent.",
                **gaze_metrics,
            }
        }