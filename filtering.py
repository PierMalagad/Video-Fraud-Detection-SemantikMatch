import os
import cv2
import pickle
import json
import logging
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class FrameData:
    """
    Data structure for cached frame information
    
    Stores all relevant information about a processed video frame containing a detected face
    This includes the face location, extracted face image, quality metrics, and frame position
    """
    bbox: Tuple[int, int, int, int] # Face bounding box: x, y, width, height in pixels
    face_crop: np.ndarray           # Normalized face image (224x224x3)
    quality_score: float            # Combined confidence and size quality score (0.0-1.0)
    frame_idx: int                  # Original frame index in the source video

@dataclass
class VideoMetadata:
    """
    Metadata for processed video containing summary information
    
    Stores high-level information about the video processing results, including which frames were successfully processed and quality metrics
    """
    video_path: str                # Path to the original video file
    total_frames: int              # Total number of frames in the video
    fps: float                     # Frames per second of the original video
    valid_frames: List[int]        # List of frame indices that contain valid faces
    avg_face_size_ratio: float     # Average ratio of face area to total frame area


class VideoFilter:
    """
    Handles video preprocessing and caching for face detection and analysis
    
    This class provides a complete pipeline for processing videos to extract face data:
    - Samples frames based on video length
    - Detects faces using MediaPipe
    - Extracts and normalizes face regions
    - Caches results to avoid reprocessing
    - Filters videos based on quality criteria
    """
    
    def __init__(self, dir_names: str, dir_cache: str, filename: List[str] = None, min_face_ratio: float = 0.15, check_eligible: bool = False, cache: bool = False) -> None:
        """
        Initialize the video preprocessor with configuration settings
        
        Args:
            dir_names: Directory path where processed video names are stored (into .txt files containing the video paths)
            dir_cache: Directory path where processed video metadata is gonna be cached
            filename (List[str]): Specific filenames to process.
            min_face_ratio: Minimum ratio of face area to frame area (0.0-1.0) -> Faces smaller than this ratio will be rejected
            check_eligible: Whether to look in specific files for already eligible videos
            cache: Whether to cache the data for each eligible video
        """
        # Set up the 2 directories
        self.dir_names = Path(dir_names)
        self.dir_names.mkdir(exist_ok=True)
        self.cache_dir = Path(dir_cache)
        self.cache_dir.mkdir(exist_ok=True)

        self.filename = filename
        self.existing_paths = []        # Will store all existing video paths
        self.check_eligible = check_eligible
        self.cache = cache
        
        # Store quality threshold for face detection
        self.min_face_ratio = min_face_ratio
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,              # Full-range model for better detection at distance
            min_detection_confidence=0.4    # Moderate confidence threshold
        )
        
        # Setup logging for debugging and monitoring
        logging.basicConfig(level=logging.INFO)

    def extract_paths(self) -> List[str]:
        """
        Extract video paths from text files in the specified directory
        
        Reads only the specified .txt files, then extracts video paths from each file
        
        Returns:
            List[str]: List of all video paths found in the text files
            
        Raises:
            FileNotFoundError: If the specified directory doesn't exist
            PermissionError: If there are permission issues accessing the directory
        """
        if self.filename is None:
            raise ValueError('No filenames to process provided')

        try:
            # Process only the specified filenames that end with .txt
            paths = [os.path.join(self.dir_names, name) for name in self.filename if name.endswith('.txt')]
            
            # Read video paths from each text file
            for path in paths:
                self.existing_paths.extend(self.read_file(path))
            
            return self.existing_paths
            
        except FileNotFoundError:
            print(f"Error: Directory '{self.dir_names}' not found")
            return []
        except PermissionError:
            print(f"Error: Permission denied accessing directory '{self.dir_names}'")
            return []
        except Exception as e:
            print(f"Unexpected error extracting paths: {e}")
            return []
    
    def read_file(self, path: str) -> List[str]:
        """
        Read a single text file and return its contents as a list of strings
        
        Each line in the file is treated as a separate video path
        Empty lines and whitespace are stripped
        
        Args:
            path (str): Full path to the text file to read
            
        Returns:
            List[str]: List of video paths from the file (one per line)
        """
        try:
            with open(path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                # Strip whitespace and remove empty lines
                return [line.strip() for line in lines if line.strip()]
        except FileNotFoundError:
            print(f"Error: File not found: {path}")
            return []
        except PermissionError:
            print(f"Error: Permission denied reading file: {path}")
            return []
        except UnicodeDecodeError:
            print(f"Error: Unable to decode file (encoding issue): {path}")
            return []
        except Exception as e:
            print(f"Error reading file {path}: {e}")
            return []
    

    def _calculate_sample_indices(self, total_frames: int, fps: float) -> List[int]:
        """
        Calculate which frames to sample based on video length and frame rate
        
        Implements adaptive sampling strategy:
        - Short videos (≤10s): ~3 FPS sampling for detailed analysis
        - Medium videos (≤30s): ~2 FPS sampling for balanced coverage
        - Long videos (>30s): ~1 FPS sampling for efficiency
        
        Ensures minimum 8 frames and maximum 30 frames regardless of video length
        
        Args:
            total_frames: Total number of frames in the video
            fps: Frames per second of the video
            
        Returns:
            List of frame indices to sample, sorted in ascending order
        """
        # Calculate video duration in seconds
        video_duration = total_frames / fps
        
        # Determine sampling step based on video length
        if video_duration <= 10:
            # Short videos: sample every 10 frames or ~3 FPS, whichever is larger
            step = max(10, int(fps * 0.33))  # ~3 FPS sampling
        elif video_duration <= 30:
            # Medium videos: sample every 15 frames or ~2 FPS, whichever is larger
            step = max(15, int(fps * 0.5))   # ~2 FPS sampling
        else:
            # Long videos: sample every 30 frames or ~1 FPS, whichever is larger
            step = max(30, int(fps * 1.0))   # ~1 FPS sampling
        
        # Generate initial sampling indices
        indices = list(range(0, total_frames, step))
        
        # Ensure we have at least 8 frames for meaningful analysis
        if len(indices) < 8:
            # If too few frames, reduce step size to get more samples
            step = max(1, total_frames // 8)
            indices = list(range(0, total_frames, step))
        elif len(indices) > 30:
            # If too many frames, increase step size to reduce processing time
            step = total_frames // 30
            indices = list(range(0, total_frames, step))
        
        # Cap at maximum 30 frames to control processing time
        return indices[:30]
    
    def _detect_face_in_frame(self, frame: np.ndarray) -> Optional[Tuple[Tuple[int, int, int, int], float]]:
        """
        Detect the best face in a single video frame using MediaPipe
        
        Processes the frame to find faces and returns the most confident detection along with a quality score that combines detection confidence and face size
        
        Args:
            frame: Input video frame as BGR numpy array
            
        Returns:
            Tuple of (bbox, quality_score) where:
            - bbox: (x, y, width, height) face bounding box in pixels
            - quality_score: Combined confidence and size quality metric
            Returns None if no suitable face is detected
        """
        try:
            # Get frame dimensions for coordinate conversion
            height, width = frame.shape[:2]
            
            # Convert BGR to RGB for MediaPipe processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run face detection
            results = self.face_detection.process(frame_rgb)
            
            # Check if any faces were detected
            if not results.detections:
                return None
            
            # Select the most confident detection from all detected faces
            best_detection = max(results.detections, key=lambda x: x.score[0])
            
            # Convert normalized coordinates to pixel coordinates
            bbox = best_detection.location_data.relative_bounding_box
            x, y = int(bbox.xmin * width), int(bbox.ymin * height)
            w, h = int(bbox.width * width), int(bbox.height * height)
            
            # Ensure bounding box stays within frame boundaries
            x, y = max(0, x), max(0, y)
            w, h = min(w, width - x), min(h, height - y)
            
            # Calculate quality metrics for face assessment
            face_area = w * h
            frame_area = width * height
            size_ratio = face_area / frame_area
            
            # Combine detection confidence with size ratio for overall quality score
            quality_score = best_detection.score[0] * min(1.0, size_ratio / self.min_face_ratio)
            
            return (x, y, w, h), quality_score
            
        except Exception as e:
            # Return None if face detection fails for any reason
            return None
    
    def _extract_and_normalize_face(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract face region from frame and normalize to standard size
        
        Crops the face region with padding, resizes to standard dimensions (224x224), and optionally applies lighting normalization
        
        Args:
            frame: Input video frame as BGR numpy array
            bbox: Face bounding box as (x, y, width, height)
            
        Returns:
            Normalized face crop as 224x224x3 numpy array
            
        Raises:
            ValueError: If the extracted face region is invalid
        """
        try:
            x, y, w, h = bbox
            
            # Add 10% padding around face for better context
            pad_x = int(w * 0.1)
            pad_y = int(h * 0.1)
            
            # Calculate padded bounding box coordinates
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(frame.shape[1], x + w + pad_x)
            y2 = min(frame.shape[0], y + h + pad_y)
            
            # Extract face region with padding
            face_crop = frame[y1:y2, x1:x2]
            
            # Validate extracted region
            if face_crop.size == 0:
                raise ValueError("Extracted face region is empty")
            
            # Resize to standard size (224x224) for consistent model input (commonly used in computer vision models)
            face_normalized = cv2.resize(face_crop, (224, 224))
            
            return face_normalized
            
        except Exception as e:
            raise ValueError(f"Failed to extract and normalize face: {e}")

    
    def save(self, video_paths: List[str], eligible: bool) -> None:
        """
        Save processed video path to a .txt file
        
        Saves video path as str and handles errors if save fails
        
        Args:
            video_paths: List of video paths
            eligible: Boolean showing eligibility of the videos
        """
        # Skip if no paths to save
        if not video_paths:
            return
            
        try:
            # Define the .txt file to write into
            if eligible:
                filename = 'eligible.txt'
            else:
                filename = 'non_eligible.txt'
            
            path = os.path.join(self.dir_names, filename)

            # Check if the file exists
            if os.path.exists(path):
                # Append the video path to the existing list
                with open(path, "a") as file:
                    file.write("\n" + "\n".join(video_paths))
            else:
                # Create a new file and write the video path
                with open(path, "w") as file:
                    file.write("\n".join(video_paths))

        except Exception as e:
            # Print error 
            print(f'Unable to save video paths: {e}')                
    

    def check_cache(self, video_path: str) -> bool:
        """
        Check if cached data exists for a video
        
        Args:
            video_path: Path to the video file
            
        Returns:
            bool: True if both frame and metadata cache files exist
        """
        try:
            filename = os.path.basename(video_path)
            name = os.path.splitext(filename)[0]
            path = os.path.join(self.cache_dir, name)
            
            return (os.path.exists(f'{path}_frames.pkl') and os.path.exists(f'{path}_metadata.json'))
        
        except Exception:
            return False
    
    def extract_cache(self, video_path: str) -> Tuple[Dict[int, FrameData], VideoMetadata]:
        """
        Extract cached data for a video

        Args:
            video_path: Path to the video file
        
        Returns:
            Tuple of (frames_data, metadata) where:
            - frames_data: Dictionary mapping frame indices to FrameData objects
            - metadata: VideoMetadata object with processing summary
        """
        try:
            filename = os.path.basename(video_path)
            name = os.path.splitext(filename)[0]
            path = os.path.join(self.cache_dir, name)

            frame_name = f'{path}_frames.pkl'
            metadata_name = f'{path}_metadata.json'

            # Load frame data
            frames_data = None
            with open(frame_name, 'rb') as f:
                frames_data = pickle.load(f)
            
            # Load metadata
            metadata = None
            with open(metadata_name, 'r') as f:
                metadata_dict = json.load(f)
                metadata = VideoMetadata(**metadata_dict)
            
            return frames_data, metadata
            
        except Exception as e:
            print(f'Error while loading cached data for video {os.path.basename(video_path)}: {e}')
            return None, None
        

    def save_cache(self, video_path: str, frames, metadata) -> None:
        """
        Save frame data and metadata to cache files
        
        Args:
            video_path: Path to the video file
            frames: Frame data to cache
            metadata: Metadata to cache
        """
        try:
            filename = os.path.basename(video_path)
            name = os.path.splitext(filename)[0]
            path = os.path.join(self.cache_dir, name)

            frame_name = f'{path}_frames.pkl'
            metadata_name = f'{path}_metadata.json'

            # Save frame data
            with open(frame_name, 'wb') as f:
                pickle.dump(frames, f)
            
            # Save metadata
            with open(metadata_name, 'w') as f:
                json.dump(metadata.__dict__, f, indent=2)
                
        except Exception as e:
            print(f'Error while saving cache for video {os.path.basename(video_path)}: {e}')
            

    def process_video(self, video_path: str) -> Tuple[Dict[int, FrameData], VideoMetadata]:
        """
        Process a video file to extract face data from sampled frames
        
        This is the main processing method that:
        1. Checks if video path exists
        2. Opens and analyzes the video file
        3. Samples frames based on video length
        4. Detects faces in each sampled frame
        5. Extracts and normalizes face regions
        
        Args:
            video_path: Path to the video file to process
            
        Returns:
            Tuple of (frames_data, metadata) where:
            - frames_data: Dictionary mapping frame indices to FrameData objects
            - metadata: VideoMetadata object with processing summary
            
        Raises:
            ValueError: If the video file cannot be opened or processed
            FileNotFoundError: If the video file does not exist
        """
        # Verify video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video file for processing
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        try:
            # Get basic video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Validate video properties
            if total_frames <= 0 or fps <= 0:
                raise ValueError(f"Invalid video properties: frames={total_frames}, fps={fps}")
            
            # Calculate which frames to sample based on video length
            sample_indices = self._calculate_sample_indices(total_frames, fps)
            
            # Initialize storage for processing results
            frames_data = {}
            valid_frames = []
            face_size_ratios = []
            
            # Process each sampled frame
            for frame_idx in sample_indices:
                try:
                    # Seek to the specific frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    
                    # Skip if frame reading failed
                    if not ret:
                        continue
                    
                    # Rotate frame 90° counterclockwise
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    
                    # Detect face in the current frame
                    detection_result = self._detect_face_in_frame(frame)
                    if detection_result is None:
                        continue
                    
                    bbox, quality_score = detection_result
                    
                    # Check if face is large enough relative to frame
                    frame_area = frame.shape[0] * frame.shape[1]
                    face_area = bbox[2] * bbox[3]
                    size_ratio = face_area / frame_area
                    
                    # Skip faces that are too small
                    if size_ratio < self.min_face_ratio:
                        continue
                    
                    # Extract and normalize the face region
                    try:
                        face_crop = self._extract_and_normalize_face(frame, bbox)
                    except ValueError:
                        # Continue if face extraction fails
                        continue
                    
                    # Store the processed frame data
                    frames_data[frame_idx] = FrameData(
                        bbox=bbox,
                        face_crop=face_crop,
                        quality_score=quality_score,
                        frame_idx=frame_idx
                    )
                    
                    # Track successful processing
                    valid_frames.append(frame_idx)
                    face_size_ratios.append(size_ratio)
                    
                except Exception as e:
                    # Continue processing other frames if one fails
                    continue
            
        finally:
            # Ensure video capture is always released
            cap.release()
        
        # Calculate average face size ratio for quality assessment
        avg_face_size_ratio = np.mean(face_size_ratios) if face_size_ratios else 0.0
        
        # Create metadata object with processing summary
        metadata = VideoMetadata(
            video_path=video_path,
            total_frames=total_frames,
            fps=fps,
            valid_frames=valid_frames,
            avg_face_size_ratio=avg_face_size_ratio
        )
        
        return frames_data, metadata
    
    def filter_videos(self, video_paths: List[str], min_valid_frames: int = 8) -> List[str]:
        """
        Filter a list of videos based on quality criteria
        
        Processes multiple videos and returns only those that meet quality standards:
        - Must have at least min_valid_frames frames with valid faces
        - Average face size must meet the minimum ratio requirement
        
        Args:
            video_paths: List of video file paths to filter
            min_valid_frames: Minimum number of valid frames required
            
        Returns:
            List of video paths that passed quality filtering
        """
        # Validate input
        if not isinstance(video_paths, list):
            raise ValueError("video_paths must be a list")
        
        # Check for existing videos
        if self.check_eligible:
            self.existing_paths = self.extract_paths()
        
        # Initialize lists to store data
        valid_videos = []
        non_valid_videos = []
        valid_existing = []
        frames_list = []
        metadata_list = []
        
        
        # Process videos with progress tracking
        with tqdm(total=len(video_paths), desc="Filtering videos", unit="video") as pbar:
            for video_path in video_paths:
                try:
                    # Avoid processing the video if already eligible
                    if video_path in self.existing_paths:
                        valid_existing.append(video_path)
                    else:
                        # Process the video
                        frames_data, metadata = self.process_video(video_path)
                        
                        # Check if video meets quality criteria
                        if (len(frames_data) >= min_valid_frames):
                            if metadata.avg_face_size_ratio >= self.min_face_ratio:
                                valid_videos.append(video_path)
                                # Cache eligible data
                                if self.cache:
                                    frames_list.append(frames_data)
                                    metadata_list.append(metadata)
                                continue
                        
                        non_valid_videos.append(video_path)
                    
                    # Update progress bar with current counts
                    pbar.set_postfix({"✓": len(valid_videos)+len(valid_existing), "✗": len(non_valid_videos)})

                    # Update progress bar
                    pbar.update(1)

                except Exception as e:
                    # Continue processing other videos if one fails
                    print(f"Failed to process {video_path}: {e}")
                    non_valid_videos.append(video_path)
                    pbar.update(1)
                

        # Cache eligible data
        if self.cache:
            for vid, frame, mtd in zip(valid_videos, frames_list, metadata_list):
                if not self.check_cache(vid):
                    self.save_cache(vid, frame, mtd)
        
        
        # Save the paths of the processed videos
        self.save(valid_videos+valid_existing, True)
        self.save(non_valid_videos, False)

        # Print final filtering results
        print(f"Filtered {len(valid_videos)+len(valid_existing)}/{len(video_paths)} videos passed quality check")
        
        # Return the list of valid videos (including existing ones)
        return valid_videos + valid_existing
    
    def get_processed_data(self, video_path: str) -> Tuple[np.ndarray, VideoMetadata]:
        """
        Get preprocessed data for model input
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (face_sequences, metadata) where face_sequences is shape (n_frames, 224, 224, 3)
        """
        # Check if cached data is available
        if self.check_cache(video_path):
            frames_data, metadata = self.extract_cache(video_path)    
        else:
            frames_data, metadata = self.process_video(video_path)
        
        # Extract face crops in temporal order
        face_sequence = []
        for frame_idx in sorted(frames_data.keys()):
            face_crop = frames_data[frame_idx].face_crop
            face_sequence.append(face_crop)
        
        if not face_sequence:
            raise ValueError(f"No valid frames found in {video_path}")
        
        # Convert to numpy array
        face_sequences = np.array(face_sequence)
        
        return face_sequences, metadata


    def __del__(self):
        """Clean up MediaPipe resources."""
        # MediaPipe Face Detection doesn't have a close() method, so we just clean up the reference
        if hasattr(self, 'face_detection'):
            self.face_detection = None