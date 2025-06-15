import os
import cv2
import torch
import pickle
import random
import numpy as np
import pandas as pd
import seaborn as sns
import mediapipe as mp
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tensorflow.keras.metrics import Precision, Recall  # Import metrics explicitly
import warnings


class FeatureExtractorLayer(layers.Layer):
    """Custom layer to extract features from sequences using pre-trained models"""
    
    def __init__(self, base_model=None, model_name=None, **kwargs):
        super().__init__(**kwargs)
        
        if base_model is not None:
            self.base_model = base_model
            self.model_name = base_model.name
        elif model_name is not None:
            # Recreate model from name during deserialization
            self.model_name = model_name
            if 'MobileNetV3' in model_name:
                self.base_model = tf.keras.applications.MobileNetV3Large(
                    input_shape=(224, 224, 3),
                    include_top=False,
                    weights='imagenet',
                    pooling='avg'
                )
            elif 'EfficientNetB2' in model_name:
                self.base_model = tf.keras.applications.EfficientNetB2(
                    input_shape=(224, 224, 3),
                    include_top=False,
                    weights='imagenet',
                    pooling='avg'
                )
            else:
                raise ValueError(f"Unknown model name: {model_name}")
        else:
            raise ValueError("Either base_model or model_name must be provided")
            
        self.base_model.trainable = True
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]
        
        # Reshape: (batch_size, seq_length, 224, 224, 3) -> (batch_size*seq_length, 224, 224, 3)
        x_reshaped = tf.reshape(inputs, (-1, 224, 224, 3))
        
        # Extract features using the base model
        features = self.base_model(x_reshaped, training=False)       # CHECK WHAT THE TRAINING PARAMETER DOES
        
        # Reshape back to sequence: (batch_size, seq_length, feature_dim)
        feature_dim = features.shape[-1]
        features = tf.reshape(features, (batch_size, seq_length, feature_dim))
        
        return features
    
    def get_config(self):
        config = super().get_config()
        config.update({'model_name': self.model_name})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class SequenceSumLayer(layers.Layer):
    """Custom layer to sum along sequence dimension"""
    
    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
    
    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=self.axis)
    
    def get_config(self):
        config = super().get_config()
        config.update({'axis': self.axis})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


@dataclass
class ModelConfig:
    """Configuration for model architecture and training"""
    architecture: str = "efficient_lstm"  # "efficient_lstm" or "attention_cnn_lstm"
    input_shape: Tuple[int, int, int, int] = (12, 224, 224, 3)  # (sequence_length, height, width, channels)
    lstm_units: int = 128
    dropout_rate: float = 0.3
    learning_rate: float = 1e-4
    batch_size: int = 16
    epochs: int = 50
    patience: int = 10
    k_folds: int = 5

class ReadingDetectionModel:
    """Main model class for reading detection with multiple architectures"""
    
    def __init__(self, config: ModelConfig, model_save_dir: str):
        self.config = config
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(exist_ok=True)
        
        self.model = None
        self.history = None
        self.cv_results = []
        
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
    
    def _create_efficient_lstm_model(self) -> Model:
        """
        Tier 2 model: Balanced Approach (75-80% accuracy)
        MobileNetV3 backbone + LSTM for temporal modeling
        """
        # Input layer
        inputs = layers.Input(shape=(12, 224, 224, 3))
        
        # Create MobileNetV3 backbone for feature extraction
        base_model = tf.keras.applications.MobileNetV3Large(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        
        # FIXED: Replace Lambda with custom layer
        features = FeatureExtractorLayer(base_model, name='feature_extractor')(inputs)      # CHECK WHAT THIS DOES
        
        # Temporal modeling with LSTM
        x = layers.LSTM(self.config.lstm_units,
                        return_sequences=True,
                        dropout=self.config.dropout_rate,
                        recurrent_dropout=self.config.dropout_rate)(features)
        
        # Additional LSTM layer for deeper temporal understanding
        x = layers.LSTM(self.config.lstm_units // 2,
                        dropout=self.config.dropout_rate,
                        recurrent_dropout=self.config.dropout_rate)(x)
        
        # Classification head
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        outputs = layers.Dense(1, activation='sigmoid', name='reading_prediction')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='EfficientLSTM')
        return model

    def _create_attention_cnn_lstm_model(self) -> Model:
        """
        Tier 1 model: High Accuracy Priority (80-85% accuracy)
        EfficientNet-B2 backbone + BiLSTM + Attention mechanism
        """
        # Input layer
        inputs = layers.Input(shape=(12, 224, 224, 3))
        
        # Create EfficientNet-B2 backbone
        base_model = tf.keras.applications.EfficientNetB2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        
        # FIXED: Replace Lambda with custom layer
        features = FeatureExtractorLayer(base_model, name='feature_extractor')(inputs)      # CHECK WHAT THIS DOES
        
        # Bidirectional LSTM for temporal modeling
        lstm_out = layers.Bidirectional(layers.LSTM(self.config.lstm_units,
                                                    return_sequences=True,
                                                    dropout=self.config.dropout_rate,
                                                    recurrent_dropout=self.config.dropout_rate))(features)
        
        # Attention mechanism
        attention_weights = layers.Dense(1, activation='tanh')(lstm_out)
        attention_weights = layers.Softmax(axis=1)(attention_weights)
        
        # Apply attention to LSTM outputs
        attended_features = layers.Multiply()([lstm_out, attention_weights])
        
        # FIXED: Replace Lambda with custom layer
        attended_features = SequenceSumLayer(axis=1, name='sequence_sum')(attended_features)        # CHECK WHAT THIS DOES
        
        # Additional processing
        x = layers.Dense(256, activation='relu')(attended_features)
        x = layers.Dropout(self.config.dropout_rate)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid', name='reading_prediction')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='AttentionCNNLSTM')
        return model
    
    def build_model(self) -> Model:
        """Build model based on configuration"""
        if self.config.architecture == "efficient_lstm":
            self.model = self._create_efficient_lstm_model()
        elif self.config.architecture == "attention_cnn_lstm":
            self.model = self._create_attention_cnn_lstm_model()
        else:
            raise ValueError(f"Unknown architecture: {self.config.architecture}")
        
        # Define the optimizer and build the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        self.model.compile(optimizer = optimizer, loss = 'binary_crossentropy',
                           metrics = ['accuracy', Precision(name='precision'), Recall(name='recall')])
        
        return self.model
    
    def prepare_data(self, preprocessor, video_paths: List[str], labels: List[int], normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data from preprocessed videos"""
        TARGET_LENGTH = 12      # Number of frames to use for each video
        X_data = []
        y_data = []
        
        
        with tqdm(zip(video_paths, labels), total=len(video_paths), desc="Loading data") as pbar:
            for video_path, label in pbar:
                try:
                    face_sequences, metadata = preprocessor.get_processed_data(video_path)
                    
                    # Ensure we have minimum number of frames (MIGHT BE REDUNDANT given the check made with the preprocessor)
                    if face_sequences.shape[0] >= 8:

                        # Pad or truncate to consistent sequence length
                        if face_sequences.shape[0] < TARGET_LENGTH:
                            # Pad by repeating last frame
                            padding_needed = TARGET_LENGTH - face_sequences.shape[0]
                            last_frame = face_sequences[-1:].repeat(padding_needed, axis=0)
                            face_sequences = np.concatenate([face_sequences, last_frame], axis=0)

                        elif face_sequences.shape[0] > TARGET_LENGTH:
                            # Truncate by taking evenly spaced frames
                            indices = np.linspace(0, face_sequences.shape[0]-1, TARGET_LENGTH, dtype=int)
                            face_sequences = face_sequences[indices]
                        
                        # Normalize pixel values
                        if normalize:
                            face_sequences = face_sequences.astype(np.float32) / 255.0
                        else:
                            face_sequences = face_sequences.astype(np.float32)
                        
                        X_data.append(face_sequences)
                        y_data.append(label)
                        
                        pbar.set_postfix({"loaded": len(X_data)})
                    
                except Exception as e:
                    print(f"Failed to load {video_path}")
        
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        
        print(f"Prepared data shape: X={X_data.shape}, y={y_data.shape}")
        return X_data, y_data
    
    def train_with_cv(self, X_data: np.ndarray, y_data: np.ndarray) -> Dict:
        """Train model with k-fold cross-validation"""
        
        kfold = StratifiedKFold(n_splits=self.config.k_folds, shuffle=True, random_state=42)
        cv_scores = []
        fold_histories = []
        confusion_matrices = []
        
        print(f"Starting {self.config.k_folds}-fold cross-validation")
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_data, y_data)):
            # Maybe add a print here
            print(f"\n=== FOLD {fold + 1}/{self.config.k_folds} ===")
            
            # Split data
            X_train, X_val = X_data[train_idx], X_data[val_idx]
            y_train, y_val = y_data[train_idx], y_data[val_idx]
            
            # Build fresh model for this fold
            self.build_model()
            
            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=self.config.patience,
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                callbacks=callbacks,
                verbose=1, class_weight = {0: 0.6, 1: 0.4}
            )
            
            # Evaluate on validation set
            val_predictions = self.model.predict(X_val, verbose=0)
            val_pred_binary = (val_predictions > 0.5).astype(int).flatten()
            
            # Calculate metrics
            accuracy = accuracy_score(y_val, val_pred_binary)
            precision, recall, f1, _ = precision_recall_fscore_support(y_val, val_pred_binary, average='binary')
            
            # Generate confusion matrix
            cm = confusion_matrix(y_val, val_pred_binary)
            confusion_matrices.append(cm)
            
            fold_result = {
                'fold': fold + 1,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm.tolist()
            }
            
            cv_scores.append(fold_result)
            fold_histories.append(history.history)
            
            #self.logger.info(f"Fold {fold + 1} Results:")
            #self.logger.info(f"  Accuracy: {accuracy:.4f}")
            #self.logger.info(f"  Precision: {precision:.4f}")
            #self.logger.info(f"  Recall: {recall:.4f}")
            #self.logger.info(f"  F1-Score: {f1:.4f}")
            #self.logger.info(f"  Confusion Matrix:\n{cm}")
        
        # Calculate average metrics
        avg_accuracy = np.mean([score['accuracy'] for score in cv_scores])
        avg_precision = np.mean([score['precision'] for score in cv_scores])
        avg_recall = np.mean([score['recall'] for score in cv_scores])
        avg_f1 = np.mean([score['f1'] for score in cv_scores])
        
        # Plot confusion matrices for all folds
        self._plot_confusion_matrices(confusion_matrices)
        
        # Store CV results
        self.cv_results = {
            'individual_folds': cv_scores,
            'avg_accuracy': avg_accuracy,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1': avg_f1,
            'std_accuracy': np.std([score['accuracy'] for score in cv_scores]),
            'confusion_matrices': confusion_matrices
        }
        
        print(f"\n=== CROSS-VALIDATION SUMMARY ===")
        print(f"Average Accuracy: {avg_accuracy:.4f} ± {self.cv_results['std_accuracy']:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")
        print(f"Average F1-Score: {avg_f1:.4f}")
        
        return self.cv_results

    def _plot_confusion_matrices(self, confusion_matrices: List[np.ndarray]):
        """Plot confusion matrices for all CV folds"""
        n_folds = len(confusion_matrices)
        fig, axes = plt.subplots(1, n_folds, figsize=(4*n_folds, 4))
        
        if n_folds == 1:
            axes = [axes]
        
        for i, cm in enumerate(confusion_matrices):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'Fold {i+1}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
            axes[i].set_xticklabels(['Not Reading', 'Reading'])
            axes[i].set_yticklabels(['Not Reading', 'Reading'])
        
        plt.tight_layout()
        plt.savefig(self.model_save_dir / f'{self.config.architecture}_confusion_matrices.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
    
    def train_final_model(self, X_data: np.ndarray, y_data: np.ndarray) -> Model:
        """Train final model on all data"""
        print("Training final model on all available data...")
        
        # Split data for final training
        X_train, X_val, y_train, y_val = train_test_split(
            X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
        )
        
        # Build final model
        self.build_model()
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=self.config.patience,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.model_save_dir / f'best_{self.config.architecture}_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train final model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=1, class_weight = {0: 0.6, 1: 0.4}
        )
        
        return self.model
    
    def save_model(self, model_name: Optional[str] = None):
        """Save the trained model and configuration"""
        if model_name is None:
            model_name = f"{self.config.architecture}_reading_detector"
        
        # Save model
        model_path = self.model_save_dir / f"{model_name}.h5"
        self.model.save(model_path)
        
        # Save configuration
        config_path = self.model_save_dir / f"{model_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        # Save training history if available
        if self.history:
            history_path = self.model_save_dir / f"{model_name}_history.pkl"
            with open(history_path, 'wb') as f:
                pickle.dump(self.history.history, f)
        
        # Save CV results if available
        if self.cv_results:
            cv_path = self.model_save_dir / f"{model_name}_cv_results.json"
            with open(cv_path, 'w') as f:
                json.dump(self.cv_results, f, indent=2, default=str)
        
        #self.logger.info(f"Model saved to {model_path}")
        return model_path
    
    def load_model(self, model_path: str):
        """Load a saved model"""
        # FIXED: Load model with custom objects
        custom_objects = {
            'FeatureExtractorLayer': FeatureExtractorLayer,
            'SequenceSumLayer': SequenceSumLayer,
            'precision': Precision(),
            'recall': Recall()
        }
        self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        #self.logger.info(f"Model loaded from {model_path}")
        return self.model
    
    def predict(self, preprocessor, video_paths: List[str]) -> List[Dict]:
        """Make predictions on new videos"""
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load_model() first.")
        
        predictions = []
        
        #self.logger.info(f"Making predictions on {len(video_paths)} videos...")
        
        with tqdm(video_paths, desc="Predicting") as pbar:
            for video_path in pbar:
                try:
                    # Get preprocessed data
                    face_sequences, metadata = preprocessor.get_processed_data(video_path)
                    
                    # Prepare for model input
                    target_length = 12
                    if face_sequences.shape[0] < target_length:
                        padding_needed = target_length - face_sequences.shape[0]
                        last_frame = face_sequences[-1:].repeat(padding_needed, axis=0)
                        face_sequences = np.concatenate([face_sequences, last_frame], axis=0)
                    elif face_sequences.shape[0] > target_length:
                        indices = np.linspace(0, face_sequences.shape[0]-1, target_length, dtype=int)
                        face_sequences = face_sequences[indices]
                    
                    # Normalize and reshape
                    face_sequences = face_sequences.astype(np.float32) / 255.0
                    face_sequences = np.expand_dims(face_sequences, axis=0)  # Add batch dimension
                    
                    # Make prediction
                    prediction_prob = self.model.predict(face_sequences, verbose=0)[0][0]
                    prediction_binary = int(prediction_prob > 0.5)
                    
                    result = {
                        'video_path': video_path,
                        'prediction_probability': float(prediction_prob),
                        'prediction_binary': prediction_binary,
                        'prediction_label': 'reading' if prediction_binary == 1 else 'not_reading',
                        'confidence': float(max(prediction_prob, 1 - prediction_prob)),
                        'num_frames': len(metadata.valid_frames) if hasattr(metadata, 'valid_frames') else 0
                    }
                    
                    predictions.append(result)
                    pbar.set_postfix({
                        "pred": result['prediction_label'],
                        "conf": f"{result['confidence']:.3f}"
                    })
                    
                except Exception as e:
                    print(f"Failed to predict {video_path}")
                    predictions.append({
                        'video_path': video_path,
                        'error': str(e)
                    })
        
        return predictions
    
    def plot_training_history(self):
        """Plot training history"""
        if not self.history:
            print("PLOT ERROR: No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Training')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Training')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.model_save_dir / f'{self.config.architecture}_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()