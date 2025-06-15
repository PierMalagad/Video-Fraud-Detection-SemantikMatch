# Video Fraud Detection

This project is a modular pipeline for detecting scripted (fraudulent) vs. non-scripted (genuine) human behavior in video content using behavioral video features and traditional machine learning models (a deep learning addendum is available). It was developed as part of the SemantikMatch initiative to improve content integrity assessments.

## Overview

The system processes human-centric video datasets by:
- Filtering videos to retain only those with sufficient face visibility
- Extracting video features using `mediapipe` and custom logic
- Training classification models (e.g., SVM, XGBoost, Random Forest) on pre-labeled data
- Predicting fraud likelihood for new/unseen videos using cached features

## Core Workflow

The pipeline is composed of the following stages:

1. Filtering (optional): Remove videos with insufficient facial data using a face-area threshold.
2. Loading: Retrieve video paths and associated labels from `.txt` and `.json` sources.
3. Feature Extraction: Compute and cache per-video behavioral features.
4. Training: Fit and evaluate classification models on extracted features.
5. Prediction: Perform batch inference on test videos.

## Folder Structure

```
Video-Fraud-Detection-SemantikMatch/
│
├── main.py                   # Main pipeline
├── filtering.py              # Filtering logic
├── dataloader.py            # Video and label loaders
├── feature_extraction.py    # Feature extractor and caching
├── models.py                # Model training utilities
├── predict.py               # Batch inference pipeline
├── data/                    # Contains video paths and labels
└── notebook/                # Contains saved models, caching dirs, etc.
```

## Run the Pipeline

Adjust the flags in `main.py` to control the pipeline:

```python
main(
    dir_videos=...,
    face_ratio=0.1,
    files_videos=['videos1.txt', 'videos2.txt'],
    dir_labels="path/to/labels.json",
    index_labels="video_name",
    mapping_labels={"nonscripted": 0, "scripted": 1},
    dir_cache="cached_features/",
    dir_models="saved_models/",
    dir_conf_matrix="confusion_matrices/",
    default_model_name=None,
    video_tests=None,
    filter=False,
    videos_to_filter=None,
    train=True,
    predict=False
)
```

## Labels & Features

- Labels are loaded from a `.json` file and mapped to binary values:
  - `nonscripted`: 0
  - `scripted`: 1

- Features are cached using `joblib` to improve performance during repeated runs.

## Models Supported

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

Training and evaluation include standard metrics:
- Accuracy
- Precision / Recall / F1
- Confusion matrices

## Example Outputs

Confusion matrices and model artifacts are saved to:
- `notebook/Filtering/ML/confusion_matrices/`
- `notebook/Filtering/ML/saved_models/`

## Dataset

You are expected to have:
- A directory of video files
- `.txt` files listing eligible videos
- A `.json` label file (`CasualConversationsV2.json`)

## To-Do

- [ ] Improve feature extraction with temporal signals
- [ ] Integrate a GUI or web interface for uploading new videos

## License

MIT License (or your choice—can be updated)

## Author

Pier Glenn Malagad  
MSc DSBA Candidate, ESSEC Business School
