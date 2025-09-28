#!/usr/bin/env python3
"""
Flowering Dual Model - Prediction Module

This module loads the trained dual classifier models and provides
functions for making predictions on new images. The model consists of both
a flower and silique classifier, and combines their outputs to determine overall flowering status.

"""

import numpy as np
import pickle
import joblib
import os
from typing import Dict, Union, Optional

class FloweringDualModel:
    """
    Dual classifier model for flowering detection
    
    This class loads and manages two separate classifiers for flower and silique detection,
    and combines their predictions to determine overall flowering status.
    """
    
    def __init__(self):
        """
        Initialize the dual classifier model
        """
        # Model components
        self.flower_model = None
        self.silique_model = None
        self.model_info = None
        
        # Thresholds (will be loaded from model_info or set manually)
        self.flower_threshold = 0.36  # Default
        self.silique_threshold = 0.385  # Default
        
        # Model metadata
        self.flower_params = None
        self.silique_params = None
        self.performance_metrics = None
        
    def load_model(self, 
                   flower_model_path: str = 'trained_flower_classifier.joblib',
                   silique_model_path: str = 'trained_silique_classifier.joblib',
                   model_info_path: str = 'trained_dual_classifier_info.pkl',
                   verbose: bool = True) -> None:
        """
        Load the trained models and metadata
        
        Args:
            flower_model_path: Path to the trained flower classifier
            silique_model_path: Path to the trained silique classifier
            model_info_path: Path to the model metadata file
            verbose: Whether to print loading information
        """
        if verbose:
            print("Loading dual classifier models...")
        
        # Load flower classifier
        if not os.path.exists(flower_model_path):
            raise FileNotFoundError(f"Flower model not found at {flower_model_path}")
        
        self.flower_model = joblib.load(flower_model_path)
        if verbose:
            print(f"Flower classifier loaded from {flower_model_path}")
        
        # Load silique classifier
        if not os.path.exists(silique_model_path):
            raise FileNotFoundError(f"Silique model not found at {silique_model_path}")
            
        self.silique_model = joblib.load(silique_model_path)
        if verbose:
            print(f"Silique classifier loaded from {silique_model_path}")
        
        # Load model metadata
        if os.path.exists(model_info_path):
            with open(model_info_path, 'rb') as f:
                self.model_info = pickle.load(f)
            
            # Update thresholds from saved info
            self.flower_threshold = self.model_info.get('flower_threshold', self.flower_threshold)
            self.silique_threshold = self.model_info.get('silique_threshold', self.silique_threshold)
            
            # Store other metadata
            self.flower_params = self.model_info.get('flower_params')
            self.silique_params = self.model_info.get('silique_params')
            self.performance_metrics = {
                'flower_metrics': self.model_info.get('flower_metrics'),
                'silique_metrics': self.model_info.get('silique_metrics'),
                'overall_accuracy': self.model_info.get('overall_accuracy'),
                'overall_precision': self.model_info.get('overall_precision'),
                'overall_recall': self.model_info.get('overall_recall'),
                'overall_f1': self.model_info.get('overall_f1')
            }
            
            if verbose:
                print(f"Model metadata loaded from {model_info_path}")
        else:
            if verbose:
                print(f"Model metadata file not found at {model_info_path}")
                print("Using default thresholds")
        
        if verbose:
            print(f"Using thresholds: Flower={self.flower_threshold:.4f}, Silique={self.silique_threshold:.4f}")
            print("Models loaded successfully!")
    
    # Keep the old method for backward compatibility
    def load_models(self, verbose: bool = True) -> None:
        """
        Load models using default paths (for backward compatibility)
        
        Args:
            verbose: Whether to print loading information
        """
        self.load_model(verbose=verbose)
    
    def set_thresholds(self, flower_threshold: float, silique_threshold: float) -> None:
        """
        Manually set the classification thresholds
        
        Args:
            flower_threshold: Threshold for flower classifier (0.0 to 1.0)
            silique_threshold: Threshold for silique classifier (0.0 to 1.0)
        """
        if not (0.0 <= flower_threshold <= 1.0):
            raise ValueError("Flower threshold must be between 0.0 and 1.0")
        if not (0.0 <= silique_threshold <= 1.0):
            raise ValueError("Silique threshold must be between 0.0 and 1.0")
            
        self.flower_threshold = flower_threshold
        self.silique_threshold = silique_threshold
        print(f"Thresholds updated: Flower={self.flower_threshold:.4f}, Silique={self.silique_threshold:.4f}")
    
    def predict_flowers(self, embedding: np.ndarray) -> Dict[str, Union[float, int]]:
        """
        Predict flower presence for a single embedding
        
        Args:
            embedding: Image embedding vector (1D or 2D numpy array)
            
        Returns:
            Dictionary with flower prediction results
        """
        if self.flower_model is None:
            raise RuntimeError("Flower model not loaded. Call load_models() first.")
        
        # Ensure embedding is 2D for sklearn
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        # Get probabilities
        flower_probs = self.flower_model.predict_proba(embedding)[0]
        flower_prob = flower_probs[1]  # Probability of flowers being present
        
        # Apply threshold
        flower_pred = 1 if flower_prob >= self.flower_threshold else 0
        
        return {
            'probability': flower_prob,
            'prediction': flower_pred,
            'present': flower_pred == 1,
            'threshold_used': self.flower_threshold
        }
    
    def predict_siliques(self, embedding: np.ndarray) -> Dict[str, Union[float, int]]:
        """
        Predict silique presence for a single embedding
        
        Args:
            embedding: Image embedding vector (1D or 2D numpy array)
            
        Returns:
            Dictionary with silique prediction results
        """
        if self.silique_model is None:
            raise RuntimeError("Silique model not loaded. Call load_models() first.")
        
        # Ensure embedding is 2D for sklearn
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        # Get probabilities
        silique_probs = self.silique_model.predict_proba(embedding)[0]
        silique_prob = silique_probs[1]  # Probability of siliques being present
        
        # Apply threshold
        silique_pred = 1 if silique_prob >= self.silique_threshold else 0
        
        return {
            'probability': silique_prob,
            'prediction': silique_pred,
            'present': silique_pred == 1,
            'threshold_used': self.silique_threshold
        }
    
    def predict(self, embedding: np.ndarray) -> Dict[str, Union[float, int, bool, str]]:
        """
        Make predictions for flower and silique presence
        
        Args:
            embedding: Image embedding vector (1D or 2D numpy array)
            
        Returns:
            Dictionary with flower and silique prediction results
        """
        if self.flower_model is None or self.silique_model is None:
            raise RuntimeError("Models not loaded. Call load_model() first.")
        
        # Get individual predictions
        flower_result = self.predict_flowers(embedding)
        silique_result = self.predict_siliques(embedding)
        
        return {
            # Individual component predictions
            'flowers': {
                'probability': flower_result['probability'],
                'prediction': flower_result['prediction'],
                'present': flower_result['present'],
                'threshold': flower_result['threshold_used']
            },
            'siliques': {
                'probability': silique_result['probability'], 
                'prediction': silique_result['prediction'],
                'present': silique_result['present'],
                'threshold': silique_result['threshold_used']
            },
            
            # Metadata
            'thresholds_used': {
                'flower': self.flower_threshold,
                'silique': self.silique_threshold
            }
        }
    
    def predict_batch(self, embeddings: np.ndarray) -> list:
        """
        Make predictions for multiple embeddings
        
        Args:
            embeddings: Array of image embeddings (2D numpy array, one embedding per row)
            
        Returns:
            List of prediction dictionaries, one for each embedding
        """
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array (n_samples, n_features)")
        
        results = []
        for i, embedding in enumerate(embeddings):
            try:
                result = self.predict(embedding)
                result['sample_index'] = i
                results.append(result)
            except Exception as e:
                print(f"Error processing sample {i}: {str(e)}")
                results.append({
                    'sample_index': i,
                    'error': str(e),
                    'flowering_status': 'Error',
                    'flowering_prediction': -1
                })
        
        return results
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded models
        
        Returns:
            Dictionary with model metadata and performance metrics
        """
        if self.model_info is None:
            return {
                'flower_threshold': self.flower_threshold,
                'silique_threshold': self.silique_threshold,
                'models_loaded': self.flower_model is not None and self.silique_model is not None,
                'metadata_available': False
            }
        
        return {
            'flower_threshold': self.flower_threshold,
            'silique_threshold': self.silique_threshold,
            'flower_params': self.flower_params,
            'silique_params': self.silique_params,
            'performance_metrics': self.performance_metrics,
            'models_loaded': self.flower_model is not None and self.silique_model is not None,
            'metadata_available': True
        }
    
    def print_model_summary(self) -> None:
        """Print a summary of the loaded models and their performance"""
        print("=" * 80)
        print("FLOWERING DUAL MODEL SUMMARY")
        print("=" * 80)
        
        info = self.get_model_info()
        
        print(f"\nModel Status:")
        print(f"  Models loaded: {'Yes' if info['models_loaded'] else 'No'}")
        print(f"  Metadata available: {'Yes' if info['metadata_available'] else 'No'}")
        
        print(f"\nClassification Thresholds:")
        print(f"  Flower threshold: {self.flower_threshold:.4f}")
        print(f"  Silique threshold: {self.silique_threshold:.4f}")
        
        print(f"\nClassification Rule:")
        print(f"  Flowering = (flowers=1 AND siliques=0)")
        print(f"  Non-Flowering = all other combinations")
        
        if info['metadata_available'] and self.performance_metrics:
            print(f"\nPerformance Metrics (from training):")
            
            flower_metrics = self.performance_metrics.get('flower_metrics', {})
            silique_metrics = self.performance_metrics.get('silique_metrics', {})
            
            print(f"  Flower Classifier:")
            print(f"    Accuracy: {flower_metrics.get('accuracy', 'N/A'):.4f}")
            print(f"    Precision: {flower_metrics.get('precision', 'N/A'):.4f}")
            print(f"    Recall: {flower_metrics.get('recall', 'N/A'):.4f}")
            print(f"    F1-Score: {flower_metrics.get('f1', 'N/A'):.4f}")
            
            print(f"  Silique Classifier:")
            print(f"    Accuracy: {silique_metrics.get('accuracy', 'N/A'):.4f}")
            print(f"    Precision: {silique_metrics.get('precision', 'N/A'):.4f}")
            print(f"    Recall: {silique_metrics.get('recall', 'N/A'):.4f}")
            print(f"    F1-Score: {silique_metrics.get('f1', 'N/A'):.4f}")
            
            print(f"  Overall Flowering Classification:")
            print(f"    Accuracy: {self.performance_metrics.get('overall_accuracy', 'N/A'):.4f}")
            print(f"    Precision: {self.performance_metrics.get('overall_precision', 'N/A'):.4f}")
            print(f"    Recall: {self.performance_metrics.get('overall_recall', 'N/A'):.4f}")
            print(f"    F1-Score: {self.performance_metrics.get('overall_f1', 'N/A'):.4f}")


# Example usage
if __name__ == "__main__":
    # Example: Load models and make a prediction
    print("Flowering Dual Model - Example Usage")
    print("=" * 50)
    
    try:
        # Load the model
        model = FloweringDualModel()
        model.load_model()
        
        # Print model summary
        model.print_model_summary()
        
        # Example: Set custom thresholds
        print(f"\nSetting custom thresholds...")
        model.set_thresholds(flower_threshold=0.4, silique_threshold=0.35)
        
        print(f"\nModel is ready for predictions!")
        print(f"Use model.predict(embedding) to classify new images.")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Make sure you have run train_test.py first to create the model files.")
        
    except Exception as e:
        print(f"Unexpected error: {e}")