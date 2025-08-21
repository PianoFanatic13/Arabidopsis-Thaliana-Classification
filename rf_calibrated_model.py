# rf_calibrated_model.py
"""
Clean and simplified Random Forest Calibrated Model for Arabidopsis Flowering Classification

This module provides a complete pipeline for training, evaluating, and predicting with a 
Random Forest model for binary classification of flowering vs non-flowering plants.

Labels: FL = 1 (flowering), Non-FL = 0 (non-flowering)
"""

import os
import numpy as np
import pandas as pd
import pickle
import joblib
from typing import Dict, Tuple, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# ========================================
# Helper Functions
# ========================================

def encode_labels(y, positive_label="FL"):
    """Convert string labels to binary (FL=1, Non-FL=0)"""
    y = np.array(y)
    return (y == positive_label).astype(int)


def load_embedding_dict(pkl_path):
    """Load embedding dictionary from pickle file"""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def build_matrix(emb_dict, df, image_col="Image #"):
    """Build feature matrix from embeddings and dataframe"""
    rows, fnames = [], []
    for img_num in df[image_col]:
        key = os.path.join("images", f"image_{img_num}.jpg")
        emb = emb_dict.get(key)
        if emb is None:
            raise KeyError(f"Embedding not found for {key}")
        rows.append(emb)
        fnames.append(key)
    return np.vstack(rows), fnames


def plot_confusion_matrix_fl_first(cm, class_names=("FL", "Non-FL"), normalize=False, title="Confusion Matrix"):
    """Plot confusion matrix with FL-first ordering"""
    if normalize:
        cm_disp = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
    else:
        cm_disp = cm.astype(int)
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_disp, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(title)
    
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm_disp[i, j], fmt), 
                   ha='center', va='center', color='black', fontsize=14)
    
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


# ========================================
# Main Model Class
# ========================================

class FloweringClassifier:
    """Random Forest Classifier for Flowering vs Non-Flowering Plant Classification"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_rf = None
        self.calibrated_model = None
        self.best_params = None
        self.best_threshold = 0.5  # Default threshold
        self.class_names = ["FL", "Non-FL"]
        
        # Parameter grid optimized to prevent overfitting
        self.param_dist = {
            'n_estimators': [50, 100, 150],   # Conservative range for efficiency
            'max_depth': [3, 5, 7, 10],       # Limited depth to prevent overfitting
            'min_samples_split': [5, 10, 20], # Higher values for regularization
            'min_samples_leaf': [2, 4, 8],    # Higher values for regularization
            'max_features': ['sqrt', 'log2'],  # Feature subset for regularization
            'bootstrap': [True, False]        # Bootstrap sampling options
        }
    
    def train_and_calibrate(self, X_train, y_train, X_test, y_test, verbose=True):
        """
        Train Random Forest with hyperparameter search and calibration
        
        Args:
            X_train, y_train: Training data and labels
            X_test, y_test: Test data and labels  
            verbose: Whether to print progress
        """
        if verbose:
            print("Starting Random Forest hyperparameter search...")
        
        # Base Random Forest with balanced class weights
        rf = RandomForestClassifier(
            class_weight='balanced', 
            random_state=self.random_state
        )
        
        # Randomized search for hyperparameters
        rf_random = RandomizedSearchCV(
            rf, 
            param_distributions=self.param_dist,
            n_iter=15,  # Number of parameter combinations to try
            cv=10,      # 10-fold cross-validation for robust evaluation
            verbose=1 if verbose else 0,
            n_jobs=-1, 
            scoring='f1_weighted', 
            random_state=self.random_state
        )
        
        rf_random.fit(X_train, y_train)
        self.best_rf = rf_random.best_estimator_
        self.best_params = rf_random.best_params_
        
        if verbose:
            print(f"Best parameters: {self.best_params}")
            print("Calibrating with sigmoid regression...")
        
        # Calibrate with sigmoid regression for better probability estimates
        self.calibrated_model = CalibratedClassifierCV(
            self.best_rf, 
            method='sigmoid',  # Using sigmoid for better overconfident model calibration
            cv=10              # 10-fold cross-validation for robust calibration
        )
        self.calibrated_model.fit(X_train, y_train)
        
        # Get test predictions and find best threshold
        y_probs = self.calibrated_model.predict_proba(X_test)[:, 1]
        self.best_threshold = self._find_best_threshold(y_test, y_probs)
        
        if verbose:
            print(f"Best threshold (F1-optimized): {self.best_threshold:.3f}")
        
        return self
    
    def _find_best_threshold(self, y_true, y_probs):
        """Find threshold that maximizes F1 score"""
        thresholds = np.linspace(0, 1, 101)
        best_f1 = 0
        best_thresh = 0.5
        
        for thresh in thresholds:
            y_pred = (y_probs >= thresh).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        return best_thresh
    
    def evaluate(self, X_test, y_test, show_plots=True):
        """
        Evaluate the model and show results
        
        Args:
            X_test, y_test: Test data and labels
            show_plots: Whether to display confusion matrix plots
        
        Returns:
            dict: Evaluation metrics
        """
        if self.calibrated_model is None:
            raise ValueError("Model not trained yet. Call train_and_calibrate() first.")
        
        # Get predictions
        y_probs = self.calibrated_model.predict_proba(X_test)[:, 1]
        y_pred = (y_probs >= self.best_threshold).astype(int)
        
        # Classification report
        print("Classification Report (FL first, then Non-FL):")
        print(classification_report(
            y_test, y_pred,
            labels=[1, 0],  # FL=1 first, Non-FL=0 second
            target_names=['FL', 'Non-FL'],
            zero_division=0
        ))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix (FL-first order):")
        print(cm)
        
        if show_plots:
            plot_confusion_matrix_fl_first(
                cm, normalize=False, 
                title="Confusion Matrix (FL-first Counts)"
            )
            plot_confusion_matrix_fl_first(
                cm, normalize=True, 
                title="Confusion Matrix (FL-first Normalized)"
            )
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary', zero_division=0
        )
        roc_auc = roc_auc_score(y_test, y_probs)
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'best_threshold': self.best_threshold
        }
        
        return metrics
    
    def predict(self, embedding):
        """
        Predict on a single embedding
        
        Args:
            embedding: Single embedding vector (1D array)
            
        Returns:
            dict: prediction probability and label
        """
        if self.calibrated_model is None:
            raise ValueError("Model not trained yet. Call train_and_calibrate() first.")
        
        # Ensure embedding is 2D for sklearn
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        # Get probability
        prob_fl = self.calibrated_model.predict_proba(embedding)[0, 1]
        
        # Apply threshold
        predicted_label = 1 if prob_fl >= self.best_threshold else 0
        label_name = "FL" if predicted_label == 1 else "Non-FL"
        
        return {
            'probability_FL': prob_fl,
            'probability_NonFL': 1 - prob_fl,
            'predicted_label': predicted_label,
            'predicted_class': label_name,
            'threshold_used': self.best_threshold
        }
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.calibrated_model is None:
            raise ValueError("No trained model to save.")
        
        model_data = {
            'calibrated_model': self.calibrated_model,
            'best_params': self.best_params,
            'best_threshold': self.best_threshold,
            'random_state': self.random_state
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.calibrated_model = model_data['calibrated_model']
        self.best_params = model_data['best_params']
        self.best_threshold = model_data['best_threshold']
        self.random_state = model_data['random_state']
        print(f"Model loaded from {filepath}")


# ========================================
# Visualization Functions
# ========================================

def visualize_embeddings(X_train, X_test, y_train, y_test, method="pca", 
                        n_components=2, title="Embedding Visualization"):
    """
    Visualize embeddings using PCA or t-SNE
    
    Args:
        X_train, X_test: Training and test embeddings
        y_train, y_test: Training and test labels
        method: 'pca' or 'tsne'
        n_components: Number of dimensions (2 or 3)
        title: Plot title
    """
    # Combine data
    X_combined = np.vstack([X_train, X_test])
    y_combined = np.hstack([y_train, y_test])
    dataset_labels = ["Train"] * len(X_train) + ["Test"] * len(X_test)
    
    # Apply dimensionality reduction
    if method.lower() == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
    elif method.lower() == "tsne":
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")
    
    X_reduced = reducer.fit_transform(X_combined)
    
    # Create plot
    fig = plt.figure(figsize=(10, 8))
    
    if n_components == 2:
        # 2D plot
        colors = ['blue', 'orange']
        markers = ['o', 's']
        
        for i, dataset in enumerate(["Train", "Test"]):
            mask = np.array(dataset_labels) == dataset
            plt.scatter(
                X_reduced[mask, 0], X_reduced[mask, 1],
                c=colors[i], marker=markers[i],
                label=f"{dataset} Data", alpha=0.7, s=30
            )
        
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
    
    elif n_components == 3:
        # 3D plot
        ax = fig.add_subplot(111, projection='3d')
        colors = ['blue', 'orange']
        
        for i, dataset in enumerate(["Train", "Test"]):
            mask = np.array(dataset_labels) == dataset
            ax.scatter(
                X_reduced[mask, 0], X_reduced[mask, 1], X_reduced[mask, 2],
                c=colors[i], label=f"{dataset} Data", alpha=0.7, s=30
            )
        
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
    
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ========================================
# Complete Pipeline Function
# ========================================

def train_flowering_classifier(train_embeddings_pkl, test_embeddings_pkl,
                             train_csv, test_csv, save_model_path=None):
    """
    Complete pipeline to train flowering classifier
    
    Args:
        train_embeddings_pkl: Path to training embeddings pickle
        test_embeddings_pkl: Path to test embeddings pickle  
        train_csv: Path to training CSV file
        test_csv: Path to test CSV file
        save_model_path: Optional path to save trained model
    
    Returns:
        FloweringClassifier: Trained classifier
        dict: Evaluation metrics
    """
    print("Loading data...")
    
    # Load embeddings and labels
    train_emb = load_embedding_dict(train_embeddings_pkl)
    test_emb = load_embedding_dict(test_embeddings_pkl)
    train_df = pd.read_csv(train_csv, dtype=str)
    test_df = pd.read_csv(test_csv, dtype=str)
    
    # Build matrices
    X_train, _ = build_matrix(train_emb, train_df)
    y_train = encode_labels(train_df["Classification"].values)
    X_test, _ = build_matrix(test_emb, test_df)
    y_test = encode_labels(test_df["Classification"].values)
    
    print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")
    print(f"Train FL ratio: {y_train.mean():.3f}, Test FL ratio: {y_test.mean():.3f}")
    
    # Train classifier
    classifier = FloweringClassifier(random_state=42)
    classifier.train_and_calibrate(X_train, y_train, X_test, y_test)
    
    # Evaluate
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    metrics = classifier.evaluate(X_test, y_test, show_plots=True)
    
    # Visualize embeddings
    print("\nGenerating embedding visualizations...")
    visualize_embeddings(X_train, X_test, y_train, y_test, 
                        method="pca", title="PCA Visualization")
    visualize_embeddings(X_train, X_test, y_train, y_test, 
                        method="tsne", title="t-SNE Visualization")
    
    # Save model if requested
    if save_model_path:
        classifier.save_model(save_model_path)
    
    return classifier, metrics


# ========================================
# Example Usage
# ========================================

if __name__ == "__main__":
    # Example usage
    classifier, metrics = train_flowering_classifier(
        "train_embeddings.pkl",
        "test_embeddings.pkl", 
        "Training Set.csv",
        "Testing Set.csv",
        save_model_path="flowering_classifier.joblib"
    )
    
    # Example prediction
    print("\n" + "="*50)
    print("EXAMPLE PREDICTION")
    print("="*50)
    
    # Load a sample embedding for demonstration
    test_emb = load_embedding_dict("test_embeddings.pkl")
    sample_key = list(test_emb.keys())[0]
    sample_embedding = test_emb[sample_key]
    
    prediction = classifier.predict(sample_embedding)
    print(f"Sample image: {sample_key}")
    print(f"Prediction: {prediction}")