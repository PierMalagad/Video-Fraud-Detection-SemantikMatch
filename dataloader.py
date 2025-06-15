import os
import joblib
import numpy as np
import tqdm as tqdm
import pandas as pd
from typing import List, Dict, Optional, Union


class VideoLoader():
    """
    A class to load video file paths from text files containing video references.
    
    This class reads text files from a specified directory and extracts video paths
    from those files. Each text file should contain video paths, one per line.
    """
    
    def __init__(self, directory: str, filename: Optional[List[str]] = None):
        """
        Initialize the VideoLoader with directory and optional specific filenames.
        
        Args:
            directory (str): Path to the directory containing text files with video paths
            filename (Optional[List[str]]): Specific filenames to process. If None, processes all .txt files in the directory
        """
        self.directory = directory
        self.filename = filename
        self.video_paths = []  # Will store all extracted video paths
    
    def extract_paths(self) -> List[str]:
        """
        Extract video paths from text files in the specified directory.
        
        Reads either all .txt files in the directory or only the specified filenames,
        then extracts video paths from each file.
        
        Returns:
            List[str]: List of all video paths found in the text files
            
        Raises:
            FileNotFoundError: If the specified directory doesn't exist
            PermissionError: If there are permission issues accessing the directory
        """
        try:
            # Determine which files to process based on initialization parameters
            if self.filename is None:
                # Process all .txt files in the directory
                paths = [os.path.join(self.directory, name) 
                        for name in os.listdir(self.directory) 
                        if name.endswith('.txt')]
            else:
                # Process only the specified filenames that end with .txt
                paths = [os.path.join(self.directory, name) 
                        for name in self.filename 
                        if name.endswith('.txt')]
            
            # Read video paths from each text file
            for path in paths:
                self.video_paths.extend(self.read_file(path))
            
            return self.video_paths
            
        except FileNotFoundError:
            print(f"Error: Directory '{self.directory}' not found")
            return []
        except PermissionError:
            print(f"Error: Permission denied accessing directory '{self.directory}'")
            return []
        except Exception as e:
            print(f"Unexpected error extracting paths: {e}")
            return []

    def read_file(self, path: str) -> List[str]:
        """
        Read a single text file and return its contents as a list of strings.
        
        Each line in the file is treated as a separate video path. Empty lines
        and whitespace are stripped.
        
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

    def load_videos(self) -> List[str]:
        """
        Main method to load all video paths from the configured text files.
        
        This is a convenience method that calls extract_paths() and returns
        the complete list of video paths.
        
        Returns:
            List[str]: Complete list of all video paths found
        """
        return self.extract_paths()


class LabelLoader():
    """
    A class to load and map video labels from a JSON file.
    
    This class reads video metadata from a JSON file, extracts label information,
    and maps those labels to desired values using a provided mapping dictionary.
    """
    
    def __init__(self, video_paths: List[str], path_label_file: str, 
                 index_name: str, mapping: Dict[str, Union[str, int]]):
        """
        Initialize the LabelLoader with video paths and label configuration.
        
        Args:
            video_paths (List[str]): List of video file paths to process
            path_label_file (str): Path to the JSON file containing video labels/metadata
            index_name (str): Column name to use as index when processing the label file
            mapping (Dict[str, Union[str, int]]): Dictionary mapping original label values 
                                                to desired output values
        """
        self.video_paths = video_paths
        self.path_label_file = path_label_file
        self.index_name = index_name
        self.mapping = mapping
    
    def read_label_file(self) -> pd.DataFrame:
        """
        Read the JSON label file and extract relevant columns.
        
        Reads the JSON file containing video metadata and returns only the
        'video_name' and 'video_setup' columns which contain the labeling information.
        
        Returns:
            pd.DataFrame: DataFrame with 'video_name' and 'video_setup' columns
            
        Raises:
            FileNotFoundError: If the label file doesn't exist
            ValueError: If the JSON file is malformed or missing required columns
        """
        try:
            # Read JSON file and extract only the columns we need for labeling
            df = pd.read_json(self.path_label_file)
            
            # Verify required columns exist
            required_columns = ['video_name', 'video_setup']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in label file: {missing_columns}")
            
            return df[['video_name', 'video_setup']]
            
        except FileNotFoundError:
            print(f"Error: Label file not found: {self.path_label_file}")
            return pd.DataFrame()
        except ValueError as e:
            print(f"Error: Invalid JSON file or missing columns: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error reading label file: {e}")
            return pd.DataFrame()
    
    def load_labels(self) -> Dict[str, Union[str, int, None]]:
        """
        Load video labels and apply mapping transformations.
        
        Reads the label file, sets the specified column as index, extracts label
        information from the 'video_setup' field, and applies the provided mapping
        to transform label values.
        
        Returns:
            Dict[str, Union[str, int, None]]: Dictionary mapping video names to 
                                            transformed label values. Returns None 
                                            for unmapped values.
            
        Notes:
            - Expects 'video_setup' to be a dictionary with a 'type' key
            - Uses the mapping dictionary to transform original label values
            - Returns None for labels that don't exist in the mapping
        """
        try:
            # Read the label file
            labels_file = self.read_label_file()
            
            # Return empty dict if file reading failed
            if labels_file.empty:
                return {}
            
            # Set the specified column as index for easier lookup
            labels_file.set_index(self.index_name, inplace=True)

            # Get all video names from the index
            names = list(labels_file.index)
            
            # Extract the 'type' field from the 'video_setup' column for each video
            # video_setup is expected to be a dictionary containing setup information
            labels = {}
            for name in names:
                try:
                    # Extract the type from video_setup dictionary
                    video_setup = labels_file.loc[name]['video_setup']
                    if isinstance(video_setup, dict) and 'type' in video_setup:
                        labels[name] = video_setup['type']
                    else:
                        print(f"Warning: Invalid video_setup format for {name}")
                        labels[name] = None
                except (KeyError, TypeError) as e:
                    print(f"Warning: Error extracting label for {name}: {e}")
                    labels[name] = None
            
            # Apply the mapping to transform original labels to desired values
            # If a label doesn't exist in mapping, it becomes None
            mapped_labels = {key: self.mapping.get(value) for key, value in labels.items()}
            
            return mapped_labels
            
        except Exception as e:
            print(f"Error loading labels: {e}")
            return {}