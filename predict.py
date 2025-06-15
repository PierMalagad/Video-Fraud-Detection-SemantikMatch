import os
import joblib
import numpy as np
from typing import Optional, List, Dict, Union, Tuple, Callable
from pathlib import Path
import logging

class VideoPredictionPipeline:
    """
    A complete ML pipeline for video classification predictions.
    Handles model loading, feature extraction, and predictions with caching.
    """
    
    def __init__(self, models_dir: str, extractor, default_model: str = None):
        """
        Initialize the prediction pipeline.
        
        Args:
            models_dir (str): Directory containing saved models
            extractor: instance of the video extractor
            default_model (str): Default model name to use
        """
        self.models_dir = Path(models_dir)
        self.extractor = extractor
        self.default_model = default_model
        self._loaded_models = {}  # Cache for loaded models
        self.class_labels = {0: "NOT READING", 1: "READING"}
        
        # Validate models directory
        if not self.models_dir.exists():
            raise FileNotFoundError(f"Models directory '{models_dir}' not found")
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def list_available_models(self) -> List[str]:
        """List all available model files in the models directory."""
        return [f.stem for f in self.models_dir.glob("*.pkl")]
    
    def load_model(self, model_name: str, force_reload: bool = False):
        """
        Load a model from disk with caching.
        
        Args:
            model_name (str): Name of the model file (without .pkl extension)
            force_reload (bool): Force reload even if cached
            
        Returns:
            Loaded scikit-learn model
        """
        # Check cache first
        if model_name in self._loaded_models and not force_reload:
            self.logger.info(f"Using cached model: {model_name}")
            return self._loaded_models[model_name]
        
        # Load from disk
        model_path = self.models_dir / f"{model_name}.pkl"
        if not model_path.exists():
            available = self.list_available_models()
            raise FileNotFoundError(f"Model '{model_name}' not found. Available models: {available}")
        
        try:
            model = joblib.load(model_path)
            self._loaded_models[model_name] = model
            self.logger.info(f"Loaded model: {model_name}")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_name}': {str(e)}")
    
    def extract_and_validate_features(self, video_path: str) -> np.ndarray:
        """
        Extract features from video and validate format.
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            np.ndarray: Extracted features ready for prediction
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        try:
            features = self.extractor.extract(video_path)
            features_array = np.array(features).reshape(1, -1)
            
            if features_array.size == 0:
                raise ValueError("No features extracted from video")
                
            return features_array
        except Exception as e:
            raise RuntimeError(f"Feature extraction failed for '{video_path}': {str(e)}")
    
    def predict_single(self, video_path: str, model_name: str = None, return_probabilities: bool = True) -> Dict:
        """
        Predict the label for a single video.
        
        Args:
            video_path (str): Path to video file
            model_name (str): Model to use (uses default if None)
            return_probabilities (bool): Whether to include probabilities
            
        Returns:
            Dict: Prediction results including label, confidence, etc.
        """
        # Use default model if none specified
        if model_name is None:
            if self.default_model is None:
                available = self.list_available_models()
                if not available:
                    raise ValueError("No models available and no default set")
                model_name = available[0]  # Use first available
                self.logger.warning(f"No model specified, using: {model_name}")
            else:
                model_name = self.default_model
        
        # Load model and extract features
        model = self.load_model(model_name)
        features = self.extract_and_validate_features(video_path)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        result = {'video_path': video_path, 'model_used': model_name, 'prediction': int(prediction), 'label': self.class_labels[prediction]}
        
        # Add probabilities if requested and model supports it
        if return_probabilities and hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            result.update({'probabilities': {'NOT_READING': float(probabilities[0]), 'READING': float(probabilities[1])},
                            'confidence': float(max(probabilities))})
        
        return result
    
    def predict_batch(self, video_paths: List[str], model_name: str = None, return_probabilities: bool = True) -> List[Dict]:
        """
        Predict labels for multiple videos efficiently.
        
        Args:
            video_paths (List[str]): List of video file paths
            model_name (str): Model to use
            return_probabilities (bool): Whether to include probabilities
            
        Returns:
            List[Dict]: List of prediction results
        """

        print('Predicting on the videos...')
        results = []
        failed_predictions = []
        
        # Load model once for all predictions
        model_name = model_name or self.default_model
        if model_name is None:
            available = self.list_available_models()
            model_name = available[0] if available else None
            
        if model_name is None:
            raise ValueError("No models available")
            
        model = self.load_model(model_name)
        
        for n, video_path in enumerate(video_paths):
            try:
                result = self.predict_single(video_path, model_name, return_probabilities)
                results.append(result)
                
                # Print results
                if return_probabilities:
                    print(f'Prediction for video {n+1}: {result["label"]} - confidence: {result["confidence"]}')
                else:
                    print(f'Prediction for video {n+1}: {result["label"]}')

                # Log prediction
                label = result['label']
                confidence = result.get('confidence', 'N/A')
                self.logger.info(f"'{video_path}' -> {label} (confidence: {confidence})")
                
            except Exception as e:
                error_result = {
                    'video_path': video_path,
                    'error': str(e),
                    'prediction': None,
                    'label': 'ERROR'
                }
                results.append(error_result)
                failed_predictions.append(video_path)
                self.logger.error(f"Prediction failed for '{video_path}': {str(e)}")
        
        if failed_predictions:
            self.logger.warning(f"Failed predictions: {len(failed_predictions)}/{len(video_paths)}")
        
        return results
    
    def compare_models(self, video_path: str, model_names: List[str] = None) -> Dict:
        """
        Compare predictions from multiple models on the same video.
        
        Args:
            video_path (str): Path to video file
            model_names (List[str]): Models to compare (uses all available if None)
            
        Returns:
            Dict: Comparison results from all models
        """
        if model_names is None:
            model_names = self.list_available_models()
        
        if not model_names:
            raise ValueError("No models available for comparison")
        
        # Extract features once
        features = self.extract_and_validate_features(video_path)
        
        comparison_results = {
            'video_path': video_path,
            'models': {}
        }
        
        for model_name in model_names:
            try:
                result = self.predict_single(video_path, model_name)
                comparison_results['models'][model_name] = result
            except Exception as e:
                comparison_results['models'][model_name] = {
                    'error': str(e),
                    'prediction': None
                }
        
        return comparison_results
    
    def get_pipeline_info(self) -> Dict:
        """Get information about the pipeline state."""
        return {
            'models_directory': str(self.models_dir),
            'available_models': self.list_available_models(),
            'loaded_models': list(self._loaded_models.keys()),
            'default_model': self.default_model,
            'extractor_type': type(self.extractor).__name__
        }