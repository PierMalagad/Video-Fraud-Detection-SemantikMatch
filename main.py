# IMPORT ALL CLASSES
import os
import cv2
import joblib
import random
import numpy as np
import pandas as pd
import seaborn as sns
import mediapipe as mp
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict, Union
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


from filtering import FrameData, VideoMetadata, VideoFilter
from dataloader import VideoLoader, LabelLoader
from feature_extraction import VideoFeatureExtractor, FeatureCacheExtractor
from models import MLModelTrainer
from predict import VideoPredictionPipeline


# DEFINE THE MAIN PIPELINE
def main(dir_videos: str, face_ratio: float, files_videos: List[str], dir_labels: str, index_labels: str, mapping_labels: Dict[str, Union[str, int]],
        dir_cache: str, dir_models: str, dir_conf_matrix: str, default_model_name: str, video_tests: List[str],
        filter: bool = False, videos_to_filter: List[str] = None, train: bool = True, predict: bool = True):
    
    # FILTERING
    if filter and videos_to_filter:
        video_filter = VideoFilter(dir_videos, 'eligible.txt', face_ratio)
        video_filter.filter_videos(videos_to_filter)


    # LOAD VIDEOS
    video_loader = VideoLoader(dir_videos, files_videos)
    video_paths = video_loader.load_videos()
    
    # LOAD LABELS
    label_loader = LabelLoader(video_paths, dir_labels, index_labels, mapping_labels)
    labels_all = label_loader.load_labels()
    labels = [labels_all[name] for name in [os.path.basename(path) for path in video_paths]]

    # EXTRACT FEATURES
    extractor = VideoFeatureExtractor()
    extractor_cached = FeatureCacheExtractor(extractor = extractor.extract, cache_dir = dir_cache)
    X = extractor_cached.cache(video_paths)
    y = np.array(labels)

    # TRAIN MODEL
    if train:
        model_trainer = MLModelTrainer(X, y, model_path = dir_models, conf_matrix_path = dir_conf_matrix)
        model_trainer.train_and_evaluate()
    
    # MAKE PREDICTIONS (FOR INFERENCE ON NEW VIDEOS)
    if predict:
        pipeline = VideoPredictionPipeline(models_dir = dir_models, extractor = extractor, default_model = default_model_name)
        batch_results = pipeline.predict_batch(video_tests)





if __name__ == '__main__':

    # FILTERING
    face_ratio = 0.1            # Minimum ratio of face area to frame area

    # PATH OF ELIGIBLE VIDEOS
    filenames_videos = ['videos_model3_claude.txt', 'videos_model3_claude_1.txt']                                 # Names of the .txt files containing the videos eligible for model training
    directory_videos = '/Workspace/Master-DSBA/SemantikMatch/Fraud Detection/data/Eligible_videos/'               # Directory containing .txt files w/ eligible files

    #  VIDEO LABELS
    path_label = "/Workspace/Master-DSBA/SemantikMatch/Fraud Detection/data/Labels/CasualConversationsV2.json"      # Path of file containing the .json file with the labels of the training videos
    index = 'video_name'
    mapping = {'nonscripted' : 0, 'scripted' : 1}

    # FEATURE EXTRACTOR
    directory_caching = "/Workspace/Master-DSBA/SemantikMatch/Fraud Detection/notebook/Filtering/ML/cached_features"        # Directory to store/retrieve cached features for each video 

    # MODEL TRAINING AND PREDICTIONS
    path_models = "/Workspace/Master-DSBA/SemantikMatch/Fraud Detection/notebook/Filtering/ML/saved_models_v3"
    path_confidence_matrices = "/Workspace/Master-DSBA/SemantikMatch/Fraud Detection/notebook/Filtering/ML/confusion_matrices_v2"


    # Run the pipeline
    main(dir_videos = directory_videos,
        face_ratio = face_ratio,
        files_videos = filenames_videos,
        dir_labels = path_label,
        index_labels = index,
        mapping_labels = mapping,
        dir_cache = directory_caching,
        dir_models = path_models,
        dir_conf_matrix = path_confidence_matrices,
        default_model_name = None,
        video_tests = None,
        filter = False,
        videos_to_filter = None,
        train = False,
        predict = True)