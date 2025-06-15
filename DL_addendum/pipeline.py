# Complete pipeline example
class ReadingDetectionPipeline:
    """Complete pipeline integrating preprocessing and model training/prediction"""
    
    def __init__(self, preprocessor_config: dict = None, model_config: ModelConfig = None, model_dir: str = None):
        # Initialize preprocessor
        if preprocessor_config is None:
            preprocessor_config = {"cache_dir": "./cache", "min_face_ratio": 0.15}
        self.preprocessor = VideoPreprocessor(**preprocessor_config)
        
        # Initialize model
        if model_config is None:
            model_config = ModelConfig()
        self.model_handler = ReadingDetectionModel(model_config, model_dir)
    
    def train_pipeline(self, video_paths: List[str], labels: List[int], 
                      architecture: str = "efficient_lstm", cache_available: bool = True) -> Dict:
        
        """Complete training pipeline"""
        print('Setting up TRAINING PIPELINE...')

        # Update architecture
        self.model_handler.config.architecture = architecture
        
        # Filter and preprocess videos if no available caches
        if not cache_available:
            videos = self.preprocessor.filter_videos(video_paths)
        else:
            videos = video_paths
        
        # Step 1: Prepare data
        print("Step 1: Preparing training data...")
        X_data, y_data = self.model_handler.prepare_data(self.preprocessor, videos, labels, normalize = True)
        
        # Step 2: Cross-validation
        print("Step 2: Running cross-validation...")
        cv_results = self.model_handler.train_with_cv(X_data, y_data)
        
        # Step 3: Train final model
        print("Step 3: Training final model...")
        final_model = self.model_handler.train_final_model(X_data, y_data)
        
        # Step 4: Save model
        print("Step 5: Saving model...")
        model_path = self.model_handler.save_model()
        
        # Plot training history
        self.model_handler.plot_training_history()
        
        return {
            'cv_results': cv_results,
            'model_path': model_path,
            'valid_videos_count': len(videos),
            'architecture': architecture
        }
    
    def predict_pipeline(self, video_paths: List[str], model_path: str = None) -> List[Dict]:
        """Complete prediction pipeline"""
        print('Starting the predictions...')

        # Load model if path provided
        if model_path:
            self.model_handler.load_model(model_path)
        
        # Filter videos first
        print("Filtering videos for prediction...")
        valid_videos = self.preprocessor.filter_videos(video_paths)
        
        # Make predictions
        predictions = self.model_handler.predict(self.preprocessor, valid_videos)
        
        return predictions