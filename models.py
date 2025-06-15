import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Union, Dict, Any
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class MLModelTrainer():
    """
    A comprehensive machine learning model trainer that handles multiple classification algorithms.
    
    This class provides functionality to train, evaluate, and save multiple machine learning models
    including Logistic Regression, SVM, Random Forest, and XGBoost. It automatically handles
    data preprocessing, model training, evaluation metrics calculation, and visualization generation.
    
    Features:
    - Automatic train/test splitting with stratification
    - Feature scaling for models that require it
    - Cross-validation evaluation
    - Confusion matrix visualization
    - Model persistence (saving trained models)
    - Comprehensive evaluation metrics
    """
    
    def __init__(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series], 
                 model_path: str, conf_matrix_path: str, test_size: float = 0.2, 
                 random_state: int = 42) -> None:
        """
        Initialize the ModelTrainer with data and configuration parameters.
        
        Args:
            X: Feature matrix (input variables) - can be numpy array or pandas DataFrame
            y: Target vector (output variable) - can be numpy array or pandas Series
            model_path: Directory path where trained models will be saved
            conf_matrix_path: Directory path where confusion matrix plots will be saved
            test_size: Proportion of dataset to include in test split (default: 0.2)
            random_state: Random seed for reproducibility (default: 42)
        
        Raises:
            ValueError: If input data is empty or paths are invalid
            OSError: If directories cannot be created
        """
        try:
            # Validate input data
            if X is None or y is None:
                raise ValueError("Input data X and y cannot be None")
            
            if len(X) == 0 or len(y) == 0:
                raise ValueError("Input data cannot be empty")
            
            if len(X) != len(y):
                raise ValueError("X and y must have the same number of samples")
            
            # Store input parameters
            self.X = X
            self.y = y
            self.model_path = model_path
            self.conf_matrix_path = conf_matrix_path
            self.random_state = random_state

            # Create output directories if they don't exist
            # This ensures we have proper locations to save models and plots
            os.makedirs(self.model_path, exist_ok=True)
            os.makedirs(self.conf_matrix_path, exist_ok=True)

            # Define the machine learning models to train and evaluate
            # Each model is configured with appropriate hyperparameters
            self.models = {
                "Logistic_Regression": LogisticRegression(
                    max_iter=1000,  # Increased iterations to ensure convergence
                    random_state=self.random_state
                ),
                "SVM": SVC(
                    kernel='rbf',  # Radial Basis Function kernel for non-linear classification
                    probability=True,  # Enable probability estimates for better evaluation
                    random_state=self.random_state
                ),
                "Random_Forest": RandomForestClassifier(
                    n_estimators=100,  # Number of trees in the forest
                    random_state=self.random_state
                ),
                "XGBoost": XGBClassifier(
                    #use_label_encoder=False,  # Disable deprecated label encoder
                    eval_metric='logloss',  # Use log loss for binary classification
                    random_state=self.random_state
                )
            }

            # Initialize the feature scaler for models that require normalized features
            self.scaler = StandardScaler()
            
            # Perform initial data splitting and scaling
            self._split_and_scale(test_size)
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ModelTrainer: {str(e)}")

    def _split_and_scale(self, test_size: float) -> None:
        """
        Split the dataset into training and testing sets, then scale the features.
        
        This method performs stratified train-test split to maintain class distribution
        in both training and testing sets. It also applies StandardScaler to normalize
        features, which is important for distance-based algorithms like SVM and Logistic Regression.
        
        Args:
            test_size: Proportion of dataset to include in test split
            
        Raises:
            ValueError: If test_size is not between 0 and 1
            RuntimeError: If splitting or scaling fails
        """
        try:
            # Validate test_size parameter
            if not 0 < test_size < 1:
                raise ValueError("test_size must be between 0 and 1")
            
            # Perform stratified train-test split
            # Stratification ensures that the proportion of samples for each target class 
            # is preserved in both training and testing sets
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, 
                test_size=test_size, 
                stratify=self.y,  # Maintain class distribution
                random_state=self.random_state, 
                shuffle=True  # Shuffle data before splitting
            )
            
            # Apply feature scaling
            # StandardScaler removes the mean and scales to unit variance
            # This is crucial for algorithms sensitive to feature scales (e.g., SVM, Logistic Regression)
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)  # Use same scaling as training
            
        except Exception as e:
            raise RuntimeError(f"Failed to split and scale data: {str(e)}")

    def _is_scaled_model(self, model_name: str) -> bool:
        """
        Determine if a specific model requires feature scaling.
        
        Some machine learning algorithms are sensitive to the scale of input features
        and perform better with normalized data. This method identifies which models
        need scaled features.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            bool: True if the model requires scaled features, False otherwise
        """
        try:
            # Currently, only Logistic Regression is configured to use scaled features
            # This can be expanded to include other scale-sensitive models like SVM
            return model_name == "Logistic_Regression"
        except Exception:
            # Default to False if there's any issue
            return False

    def _save_confusion_matrix(self, model_name: str, cm: np.ndarray, acc: float, 
                             prec: float, rec: float, f1: float) -> None:
        """
        Generate and save a confusion matrix heatmap with evaluation metrics.
        
        Creates a visual representation of the model's performance using a heatmap
        that shows true vs predicted classifications. The plot includes key metrics
        in the title for quick reference.
        
        Args:
            model_name: Name of the model (used for plot title and filename)
            cm: Confusion matrix as numpy array
            acc: Accuracy score
            prec: Precision score  
            rec: Recall score
            f1: F1 score
            
        Raises:
            RuntimeError: If plot generation or saving fails
        """
        try:
            # Create a new figure with specified size
            plt.figure(figsize=(6, 5))
            
            # Generate heatmap using seaborn
            sns.heatmap(
                cm, 
                annot=True,  # Show numbers in each cell
                fmt="d",  # Format numbers as integers
                cmap="Blues",  # Color scheme
                cbar=False,  # Hide color bar for cleaner look
                xticklabels=["Negative", "Positive"],  # X-axis labels
                yticklabels=["Negative", "Positive"]   # Y-axis labels
            )
            
            # Set plot title with model name and key metrics
            plt.title(f"{model_name} Confusion Matrix\n"
                     f"Acc: {acc:.2f} | Prec: {prec:.2f} | Rec: {rec:.2f} | F1: {f1:.2f}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()  # Adjust layout to prevent label cutoff
            
            # Save the plot to file
            plot_path = os.path.join(self.conf_matrix_path, f"{model_name}_confusion_matrix.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()  # Close the figure to free memory
            
        except Exception as e:
            # Ensure plot is closed even if saving fails
            plt.close()
            raise RuntimeError(f"Failed to save confusion matrix for {model_name}: {str(e)}")

    def train_and_evaluate(self) -> None:
        """
        Train and evaluate all configured machine learning models.
        
        This is the main method that orchestrates the entire machine learning pipeline:
        1. Iterates through each configured model
        2. Trains the model on training data
        3. Makes predictions on test data
        4. Calculates evaluation metrics
        5. Performs cross-validation
        6. Saves the trained model
        7. Generates and saves confusion matrix visualization
        
        The method provides comprehensive output including confusion matrices,
        accuracy, precision, recall, F1-score, and cross-validation results.
        
        Raises:
            RuntimeError: If training or evaluation fails for any model
        """
        try:
            # Iterate through each model in the models dictionary
            for name, model in self.models.items():
                print(f"\n🧠 Training and evaluating {name}")
                
                try:
                    # Determine whether to use scaled features for this model
                    use_scaled = self._is_scaled_model(name)
                    X_tr = self.X_train_scaled if use_scaled else self.X_train
                    X_te = self.X_test_scaled if use_scaled else self.X_test

                    # Train the model on training data
                    model.fit(X_tr, self.y_train)
                    
                    # Make predictions on test data
                    y_pred = model.predict(X_te)

                    # Save the trained model to disk for later use
                    model_file_path = os.path.join(self.model_path, f"{name}.pkl")
                    joblib.dump(model, model_file_path)

                    # Calculate evaluation metrics
                    # These metrics provide different perspectives on model performance
                    acc = accuracy_score(self.y_test, y_pred)        # Overall correctness
                    prec = precision_score(self.y_test, y_pred)      # Positive prediction accuracy  
                    rec = recall_score(self.y_test, y_pred)          # Ability to find positive cases
                    f1 = f1_score(self.y_test, y_pred)              # Harmonic mean of precision and recall
                    cm = confusion_matrix(self.y_test, y_pred)       # Detailed classification breakdown

                    # Display results
                    print("   Results:")
                    print("   Confusion Matrix:")
                    print(f"   {cm}")
                    print(f"   Accuracy:  {acc:.4f}")
                    print(f"   Precision: {prec:.4f}")
                    print(f"   Recall:    {rec:.4f}")
                    print(f"   F1-Score:  {f1:.4f}")

                    # Perform cross-validation for more robust performance estimation
                    # Cross-validation provides a better estimate of how the model will perform
                    # on unseen data by training and testing on multiple data splits
                    print(f"   Performing cross-validation...")
                    X_cv = self.scaler.fit_transform(self.X) if use_scaled else self.X
                    scores = cross_val_score(model, X_cv, self.y, cv=5, scoring='accuracy')
                    print(f"\nCross-validated accuracy (5-fold): {scores.mean():.4f} ± {scores.std():.4f}")

                    # Generate and save confusion matrix visualization
                    self._save_confusion_matrix(name, cm, acc, prec, rec, f1)
                    
                except Exception as model_error:
                    print(f"   Error training {name}: {str(model_error)}")
                    continue  # Continue with next model even if current one fails
                    
        except Exception as e:
            raise RuntimeError(f"Failed to complete training and evaluation: {str(e)}")