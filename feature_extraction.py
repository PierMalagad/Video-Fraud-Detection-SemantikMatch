import os
import cv2
import joblib
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from typing import List, Tuple, Optional, Callable


class VideoFeatureExtractor():
    """
    A class to extract various eye-tracking and facial behavior features from video files.
    
    This class uses MediaPipe's Face Mesh to detect facial landmarks and extract features
    related to eye movement, blinking patterns, head pose, and concentration metrics.
    Features include eye aspect ratio, blink dynamics, saccade patterns, fixation behavior,
    and head pose angles.
    """
    
    def __init__(self):
        """
        Initialize the VideoFeatureExtractor with MediaPipe Face Mesh detector.
        
        Sets up the face mesh detector with refined landmarks for more precise
        eye and facial feature detection.
        """
        try:
            self.detector = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        except Exception as e:
            print(f"Error initializing MediaPipe Face Mesh: {e}")
            self.detector = None

    def get_eye_aspect_ratio(self, landmarks: np.ndarray, eye_indices: List[int]) -> float:
        """
        Calculate the Eye Aspect Ratio (EAR) from facial landmarks.
        
        EAR is used to detect blinks and measure eye openness. It's calculated as
        the ratio of eye height to eye width using specific landmark points.
        
        Args:
            landmarks (np.ndarray): Array of facial landmark coordinates
            eye_indices (List[int]): List of 6 landmark indices defining the eye shape [outer_corner, top_left, top_center, top_right, bottom_right, bottom_center, bottom_left, outer_corner]
        
        Returns:
            float: Eye aspect ratio value. Lower values indicate more closed eyes.
        """
        try:
            # Calculate vertical distances (eye height at different points)
            left = np.linalg.norm(landmarks[eye_indices[1]] - landmarks[eye_indices[5]])
            right = np.linalg.norm(landmarks[eye_indices[2]] - landmarks[eye_indices[4]])
            # Calculate horizontal distance (eye width)
            top = np.linalg.norm(landmarks[eye_indices[0]] - landmarks[eye_indices[3]])
            # EAR formula: (left_height + right_height) / (2 * eye_width)
            return (left + right) / (2.0 * top) if top > 0 else 0
        except (IndexError, ZeroDivisionError) as e:
            print(f"Error calculating eye aspect ratio: {e}")
            return 0

    def get_blink_rate_and_dynamics(self, ear_values: List[float], threshold: float = 0.2) -> Tuple[float, float, float]:
        """
        Analyze blink patterns from Eye Aspect Ratio values over time.
        
        Detects blinks when EAR falls below threshold and calculates various
        blink-related metrics including rate, duration, and intervals.
        
        Args:
            ear_values (List[float]): Sequence of Eye Aspect Ratio values over time
            threshold (float): EAR threshold below which a blink is detected (default: 0.2)
        
        Returns:
            Tuple[float, float, float]: 
                - blink_rate: Proportion of frames where eyes are closed
                - avg_blink_duration: Average duration of blinks in frames
                - avg_inter_blink_interval: Average time between blink starts in frames
        """
        if not ear_values:
            return 0, 0, 0
            
        try:
            blink_durations, blink_start_indices = [], []
            in_blink, blink_start_idx = False, None

            # Detect blink events and measure their durations
            for i, ear in enumerate(ear_values):
                if ear < threshold:  # Eye is closed (blink detected)
                    if not in_blink:
                        in_blink = True
                        blink_start_idx = i
                        blink_start_indices.append(i)
                else:  # Eye is open
                    if in_blink:
                        blink_durations.append(i - blink_start_idx)
                        in_blink = False

            # Handle case where video ends during a blink
            if in_blink and blink_start_idx is not None:
                blink_durations.append(len(ear_values) - blink_start_idx)

            # Calculate blink metrics
            blink_rate = sum(1 for ear in ear_values if ear < threshold) / len(ear_values)
            avg_blink_duration = np.mean(blink_durations) if blink_durations else 0
            
            # Calculate intervals between consecutive blinks
            inter_blink_intervals = np.diff(blink_start_indices) if len(blink_start_indices) > 1 else []
            avg_inter_blink_interval = np.mean(inter_blink_intervals) if len(inter_blink_intervals) > 0 else 0

            return blink_rate, avg_blink_duration, avg_inter_blink_interval
            
        except Exception as e:
            print(f"Error calculating blink dynamics: {e}")
            return 0, 0, 0

    def get_saccade_amplitude(self, eye_positions: List[np.ndarray]) -> float:
        """
        Calculate the average amplitude of saccadic eye movements.
        
        Saccades are rapid eye movements between fixation points. This method
        measures the average distance of eye movements between consecutive frames.
        
        Args:
            eye_positions (List[np.ndarray]): Sequence of eye position coordinates over time
        
        Returns:
            float: Average saccade amplitude (Euclidean distance between consecutive positions)
        """
        if len(eye_positions) < 2:
            return 0
            
        try:
            # Calculate Euclidean distance between consecutive eye positions
            diffs = [np.linalg.norm(eye_positions[i] - eye_positions[i - 1]) 
                    for i in range(1, len(eye_positions))]
            return np.mean(diffs) if diffs else 0
        except Exception as e:
            print(f"Error calculating saccade amplitude: {e}")
            return 0

    def get_fixation_features(self, eye_positions: List[np.ndarray], movement_threshold: float = 0.005) -> Tuple[int, float]:
        """
        Analyze eye fixation patterns from eye position data.
        
        Fixations are periods when the eye remains relatively stationary.
        This method identifies fixation periods and calculates related metrics.
        
        Args:
            eye_positions (List[np.ndarray]): Sequence of eye position coordinates over time
            movement_threshold (float): Maximum movement distance to be considered a fixation (default: 0.005)
        
        Returns:
            Tuple[int, float]:
                - fixation_count: Total number of distinct fixation periods
                - avg_fixation_duration: Average duration of fixations in frames
        """
        if len(eye_positions) < 2:
            return 0, 0
            
        try:
            fixation_count, current_fixation = 0, 0
            fixation_durations = []

            # Identify fixation periods based on movement threshold
            for i in range(1, len(eye_positions)):
                movement = np.linalg.norm(eye_positions[i] - eye_positions[i-1])
                if movement < movement_threshold:  # Eye is relatively stationary (fixating)
                    current_fixation += 1
                else:  # Eye moved significantly (end of fixation)
                    if current_fixation > 0:
                        fixation_durations.append(current_fixation)
                        fixation_count += 1
                        current_fixation = 0

            # Handle case where video ends during a fixation
            if current_fixation > 0:
                fixation_durations.append(current_fixation)
                fixation_count += 1

            avg_fixation_duration = np.mean(fixation_durations) if fixation_durations else 0
            return fixation_count, avg_fixation_duration
            
        except Exception as e:
            print(f"Error calculating fixation features: {e}")
            return 0, 0

    def get_head_pose_angles(self, landmarks: np.ndarray, frame_width: int, frame_height: int) -> Tuple[float, float, float]:
        """
        Calculate head pose angles (yaw, pitch, roll) from facial landmarks.
        
        Uses perspective-n-point (PnP) algorithm to estimate 3D head orientation
        from 2D facial landmark positions. This helps understand head movement patterns.
        
        Args:
            landmarks (np.ndarray): Array of facial landmark coordinates (normalized 0-1)
            frame_width (int): Width of the video frame in pixels
            frame_height (int): Height of the video frame in pixels
        
        Returns:
            Tuple[float, float, float]: Head pose angles in degrees
                - yaw: Left-right head rotation (negative=left, positive=right)
                - pitch: Up-down head rotation (negative=down, positive=up)  
                - roll: Tilt head rotation (negative=left tilt, positive=right tilt)
        """
        if len(landmarks) < 468:
            return 0, 0, 0

        try:
            # Select 6 key facial landmarks for pose estimation
            # These correspond to: nose tip, chin, left eye corner, right eye corner, 
            # left mouth corner, right mouth corner
            image_points = np.array([
                landmarks[1], landmarks[152], landmarks[33],
                landmarks[263], landmarks[61], landmarks[291]
            ], dtype="double")
            
            # Convert normalized coordinates to pixel coordinates
            image_points[:, 0] *= frame_width
            image_points[:, 1] *= frame_height

            # 3D model points for the corresponding facial features (in mm)
            model_points = np.array([
                [0.0, 0.0, 0.0],        # Nose tip
                [0.0, -63.6, -12.5],    # Chin
                [-43.3, 32.7, -26.0],   # Left eye corner
                [43.3, 32.7, -26.0],    # Right eye corner
                [-28.9, -28.9, -24.1],  # Left mouth corner
                [28.9, -28.9, -24.1]    # Right mouth corner
            ])

            # Camera calibration parameters (simplified)
            focal_length = frame_width
            center = (frame_width / 2, frame_height / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")
            dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

            # Solve PnP to get rotation vector
            success, rotation_vector, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
            if not success:
                return 0, 0, 0

            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            
            # Extract Euler angles from rotation matrix
            sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
            singular = sy < 1e-6

            if not singular:
                x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])  # Pitch
                y = np.arctan2(-rotation_matrix[2, 0], sy)                    # Yaw
                z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])  # Roll
            else:
                x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                y = np.arctan2(-rotation_matrix[2, 0], sy)
                z = 0

            # Convert from radians to degrees
            return np.degrees(y), np.degrees(x), np.degrees(z)
            
        except Exception as e:
            print(f"Error calculating head pose angles: {e}")
            return 0, 0, 0

    def get_concentration_metric(self, landmarks: np.ndarray) -> float:
        """
        Calculate a concentration metric based on eyelid distance.
        
        This metric measures the average distance between upper and lower eyelids
        for both eyes, which can be an indicator of alertness and concentration levels.
        
        Args:
            landmarks (np.ndarray): Array of facial landmark coordinates
        
        Returns:
            float: Concentration metric (higher values may indicate more alertness)
        """
        if len(landmarks) < 468:
            return 0
            
        try:
            # Calculate eyelid distances for left and right eyes
            # Using specific landmark indices for upper and lower eyelids
            left = np.linalg.norm(landmarks[65] - landmarks[159])    # Left eye eyelid distance
            right = np.linalg.norm(landmarks[295] - landmarks[386])  # Right eye eyelid distance
            return (left + right) / 2.0  # Average of both eyes
        except Exception as e:
            print(f"Error calculating concentration metric: {e}")
            return 0

    def extract(self, video_path: str) -> List[float]:
        """
        Extract comprehensive facial and eye-tracking features from a video file.
        
        Processes the entire video frame by frame, extracting various behavioral
        and physiological features that can be used for attention, engagement,
        or cognitive load analysis.
        
        Args:
            video_path (str): Path to the input video file
        
        Returns:
            List[float]: List of 12 extracted features:
                [0] mean_EAR: Average eye aspect ratio
                [1] blink_rate: Proportion of frames with blinks
                [2] avg_blink_duration: Average blink duration in frames
                [3] avg_inter_blink_interval: Average time between blinks
                [4] head_movement_variance: Variance in head position
                [5] saccade_amplitude: Average eye movement amplitude
                [6] fixation_count: Number of eye fixation periods
                [7] avg_fixation_duration: Average fixation duration
                [8] avg_yaw: Average head yaw angle (degrees)
                [9] avg_pitch: Average head pitch angle (degrees)
                [10] avg_roll: Average head roll angle (degrees)
                [11] avg_concentration: Average concentration metric
        """
        if self.detector is None:
            print("Error: MediaPipe detector not initialized")
            return [0] * 12
            
        try:
            # Initialize video capture
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video file: {video_path}")
                return [0] * 12
                
            # Initialize feature storage lists
            ear_values, nose_positions, eye_positions, fixation_eye_positions = [], [], [], []
            head_pose_angles_list, concentration_metrics = [], []

            # Get frame dimensions
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Could not read first frame from: {video_path}")
                cap.release()
                return [0] * 12
                
            frame_height, frame_width = frame.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning

            # Process each frame in the video
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                try:
                    # Convert frame to RGB for MediaPipe processing
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.detector.process(frame_rgb)
                    
                    # Extract features if face is detected
                    if results.multi_face_landmarks:
                        for face in results.multi_face_landmarks:
                            # Convert landmarks to numpy array of (x, y) coordinates
                            landmarks = np.array([(lm.x, lm.y) for lm in face.landmark])
                            
                            # Extract eye aspect ratio (for blink detection)
                            EAR = self.get_eye_aspect_ratio(landmarks, [33, 160, 158, 153, 144, 133])
                            ear_values.append(EAR)
                            
                            # Store landmark positions for movement analysis
                            nose_positions.append(landmarks[1])      # Nose tip for head movement
                            eye_positions.append(landmarks[33])      # Eye corner for saccades
                            fixation_eye_positions.append(landmarks[33])  # Same for fixation analysis
                            
                            # Calculate head pose angles
                            yaw, pitch, roll = self.get_head_pose_angles(landmarks, frame_width, frame_height)
                            head_pose_angles_list.append((yaw, pitch, roll))
                            
                            # Calculate concentration metric
                            concentration_metrics.append(self.get_concentration_metric(landmarks))
                            
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    continue
                    
            cap.release()

            # Calculate aggregate features from collected data
            mean_EAR = np.mean(ear_values) if ear_values else 0
            blink_rate, avg_blink_duration, avg_inter_blink_interval = self.get_blink_rate_and_dynamics(ear_values)
            
            # Head movement variance (stability measure)
            head_movement_variance = np.std([lm[0] for lm in nose_positions]) if nose_positions else 0
            
            # Eye movement features
            saccade_amplitude = self.get_saccade_amplitude(eye_positions)
            fixation_count, avg_fixation_duration = self.get_fixation_features(fixation_eye_positions)

            # Head pose averages
            if head_pose_angles_list:
                yaw_vals, pitch_vals, roll_vals = zip(*head_pose_angles_list)
                avg_yaw = np.mean(yaw_vals)
                avg_pitch = np.mean(pitch_vals)
                avg_roll = np.mean(roll_vals)
            else:
                avg_yaw = avg_pitch = avg_roll = 0

            # Average concentration metric
            avg_concentration = np.mean(concentration_metrics) if concentration_metrics else 0

            # Return all extracted features as a list
            return [
                mean_EAR, blink_rate, avg_blink_duration, avg_inter_blink_interval,
                head_movement_variance, saccade_amplitude,
                fixation_count, avg_fixation_duration,
                avg_yaw, avg_pitch, avg_roll,
                avg_concentration
            ]
            
        except Exception as e:
            print(f"Error extracting features from {video_path}: {e}")
            return [0] * 12


class FeatureCacheExtractor():
    """
    A caching wrapper for feature extraction to avoid recomputing features for the same videos.
    
    This class wraps any feature extractor and provides caching functionality using joblib.
    Features are cached to disk and automatically loaded if they exist, significantly
    speeding up repeated analysis of the same video files.
    """
    
    def __init__(self, extractor: Callable[[str], List[float]], cache_dir: str):
        """
        Initialize the caching feature extractor.
        
        Args:
            extractor (Callable[[str], List[float]]): Feature extraction function that takes a video path and returns a list of features
            cache_dir (str): Directory where cached features will be stored as .pkl files
        """
        self.extractor = extractor
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except Exception as e:
            print(f"Error creating cache directory {cache_dir}: {e}")

    def _safe_filename(self, path: str) -> str:
        """
        Generate a safe cache filename from a video file path.
        
        Converts the video file path to a safe filename for caching by using
        the base filename and adding a .pkl extension.
        
        Args:
            path (str): Original video file path
        
        Returns:
            str: Safe cache file path with .pkl extension
        """
        try:
            # Extract filename without extension and create cache path
            filename = os.path.basename(path)
            name, _ = os.path.splitext(filename)
            return os.path.join(self.cache_dir, f"{name}.pkl")
        except Exception as e:
            print(f"Error generating cache filename for {path}: {e}")
            # Fallback to a generic filename if path parsing fails
            return os.path.join(self.cache_dir, "unknown_file.pkl")

    def extract(self, path: str) -> List[float]:
        """
        Extract features from a video with caching support.
        
        Checks if features have been previously computed and cached. If so,
        loads from cache. Otherwise, computes features using the wrapped
        extractor and saves to cache for future use.
        
        Args:
            path (str): Path to the video file
        
        Returns:
            List[float]: List of extracted features (same as wrapped extractor)
        """
        try:
            cache_path = self._safe_filename(path)
            
            # Load from cache if it exists
            if os.path.exists(cache_path):
                try:
                    cached_features = joblib.load(cache_path)
                    return cached_features
                except Exception as e:
                    print(f"Error loading cached features for {path}: {e}")
                    # Continue to extract features if cache loading fails
            
            # Extract features using the wrapped extractor
            features = self.extractor(path)
            
            # Save features to cache for future use
            try:
                joblib.dump(features, cache_path)
            except Exception as e:
                print(f"Error saving features to cache for {path}: {e}")
                # Continue even if caching fails
            
            return features
            
        except Exception as e:
            print(f"Error in cached extraction for {path}: {e}")
            # Return default features if everything fails
            return [0] * 12

    def cache(self, video_paths: List[str]) -> np.ndarray:
        """
        Extract features from multiple videos with progress tracking and caching.
        
        Processes a list of video files, extracting features from each with
        caching support and progress visualization using tqdm.
        
        Args:
            video_paths (List[str]): List of paths to video files to process
        
        Returns:
            np.ndarray: 2D array where each row contains features for one video [Shape: (num_videos, num_features)]
            
        """
        if not video_paths:
            print("Warning: No video paths provided")
            return np.array([])
            
        try:
            all_features = []
            
            # Process each video with progress bar
            for path in tqdm(video_paths, desc='Extracting features: '):
                try:
                    features = self.extract(path)
                    all_features.append(features)
                except Exception as e:
                    print(f"Error processing {path}: {e}")
                    # Add default features if extraction fails
                    all_features.append([0] * 12)
            
            return np.array(all_features)
            
        except Exception as e:
            print(f"Error in batch feature extraction: {e}")
            return np.array([])