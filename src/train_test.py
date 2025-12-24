#!/usr/bin/env python3
"""
Dual Classifier Training and Testing Script

This script trains and tests two separate Random Forest classifiers:
1. Flower Classifier: Detects presence of flowers in images
2. Silique Classifier: Detects presence of siliques in images

The script performs hyperparameter optimization but uses customs thresholds found
to prioritize flowering precision and recall balance:
- Flower threshold: 0.36
- Silique threshold: 0.385

Overall flowering classification rule: Flowering = (flowers=1 AND siliques=0)
"""

import pandas as pd
import numpy as np
import pickle
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support,
    f1_score,
    accuracy_score
)
import warnings
warnings.filterwarnings('ignore')

def build_matrix_gbif(emb_dict, df, image_col="Image Name"):
    """Build feature matrix from embeddings and dataframe"""
    rows, fnames = [], []
    missing_count = 0
    
    for img_name in df[image_col]:
        # Expected image format: "gbifID_#"
        key = os.path.join("images", f"{img_name}.jpg")
        
        if key in emb_dict:
            rows.append(emb_dict[key])
            fnames.append(key)
        else:
            print(f"Warning: Embedding not found for {key}")
            missing_count += 1
    
    if missing_count > 0:
        print(f"WARNING: {missing_count} embeddings were missing")
    
    if len(rows) == 0:
        raise ValueError(f"No embeddings found! All {len(df)} images failed to match.")
    
    print(f"Successfully matched {len(rows)} out of {len(df)} images")
    return np.vstack(rows), fnames

def train_calibrated_classifier(X_train, y_train, classifier_name, random_state=42):
    """Train a Random Forest classifier with hyperparameter optimization and calibration"""
    
    print(f"\n{'='*60}")
    print(f"TRAINING {classifier_name.upper()} CLASSIFIER")
    print(f"{'='*60}")
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Class distribution - Class 1: {y_train.sum()}, Class 0: {len(y_train) - y_train.sum()}")
    print(f"Class balance ratio: {y_train.mean():.3f}")
    
    param_dist = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [2, 4, 8],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False]
    }
    
    # Base Random Forest with balanced class weights
    rf = RandomForestClassifier(
        class_weight='balanced',
        random_state=random_state
    )
    
    print("\nStarting hyperparameter optimization...")
    rf_random = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=15,  
        cv=10,      
        verbose=1,
        n_jobs=-1,
        scoring='f1_weighted',
        random_state=random_state
    )
    
    rf_random.fit(X_train, y_train)
    best_rf = rf_random.best_estimator_
    best_params = rf_random.best_params_
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best CV score: {rf_random.best_score_:.4f}")
    

    print("\nCalibrating with sigmoid regression...")
    calibrated_model = CalibratedClassifierCV(
        best_rf,
        method='sigmoid',
        cv=10
    )
    calibrated_model.fit(X_train, y_train)
    
    return {
        'model': calibrated_model,
        'best_params': best_params,
        'cv_score': rf_random.best_score_
    }

def evaluate_classifier(y_true, y_pred, y_probs, classifier_name, threshold):
    """Comprehensive evaluation of a binary classifier"""
    
    print(f"\n{'='*60}")
    print(f"{classifier_name.upper()} CLASSIFIER EVALUATION")
    print(f"{'='*60}")
    print(f"Threshold used: {threshold:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=['Absent', 'Present'],
        zero_division=0
    ))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                    Predicted")
    print(f"                Absent    Present")
    print(f"Actual  Absent    {cm[0,0]:4d}      {cm[0,1]:4d}")
    print(f"        Present   {cm[1,0]:4d}      {cm[1,1]:4d}")
    print(f"        ----------------------")
    print(f"        Total:    {cm.sum():4d}      (100%)")
    
    # Additional metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_probs)
    
    print(f"\nKey Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }

def main():
    print("="*80)
    print("DUAL CLASSIFIER TRAINING AND TESTING")
    print("="*80)
    
    # Load Datasets
    
    print("\n1. LOADING DATASETS...")
    
    # Load training sets
    flower_train_df = pd.read_csv("../data/Flower Training Set.csv", dtype=str)
    silique_train_df = pd.read_csv("../data/Silique Training Set.csv", dtype=str)
    test_df = pd.read_csv("../data/Testing Set New.csv", dtype=str)
    
    print(f"Flower training set: {len(flower_train_df)} samples")
    print(f"Silique training set: {len(silique_train_df)} samples")
    print(f"Testing set: {len(test_df)} samples")
    
    # Load embeddings
    print("\nLoading embeddings...")
    with open("flower_train_embeddings.pkl", "rb") as f:
        flower_train_embeddings = pickle.load(f)
    
    with open("silique_train_embeddings.pkl", "rb") as f:
        silique_train_embeddings = pickle.load(f)
    
    with open("test_embeddings_new.pkl", "rb") as f:
        test_embeddings = pickle.load(f)
    
    print(f"Flower train embeddings: {len(flower_train_embeddings)} images")
    print(f"Silique train embeddings: {len(silique_train_embeddings)} images")
    print(f"Test embeddings: {len(test_embeddings)} images")
    
    
    print("\n2. BUILDING FEATURE MATRICES...")
    
    # Build feature matrices for flower classifier
    print("\nBuilding flower classifier matrices...")
    X_flower_train, flower_train_files = build_matrix_gbif(flower_train_embeddings, flower_train_df)
    y_flower_train = flower_train_df["Classification"].astype(int).values
    
    # Build feature matrices for silique classifier  
    print("Building silique classifier matrices...")
    X_silique_train, silique_train_files = build_matrix_gbif(silique_train_embeddings, silique_train_df)
    y_silique_train = silique_train_df["Classification"].astype(int).values
    
    # Build test matrices
    print("Building test matrices...")
    X_test, test_files = build_matrix_gbif(test_embeddings, test_df)
    
    # Prepare test labels for both classifiers
    y_test_flowers = test_df["flowers"].astype(int).values  # 1 if flowers present, 0 if not
    y_test_siliques = test_df["siliques"].astype(int).values  # 1 if siliques present, 0 if not
    y_test_flowering = (test_df["flowering"] == "FL").astype(int).values  # Overall flowering status
    
    print(f"\nData shapes:")
    print(f"Flower train: {X_flower_train.shape}, labels: {len(y_flower_train)}")
    print(f"Silique train: {X_silique_train.shape}, labels: {len(y_silique_train)}")
    print(f"Test: {X_test.shape}")
    
    print(f"\nLabel distributions:")
    print(f"Flower training - Class 1: {y_flower_train.sum()}, Class 0: {len(y_flower_train) - y_flower_train.sum()}")
    print(f"Silique training - Class 1: {y_silique_train.sum()}, Class 0: {len(y_silique_train) - y_silique_train.sum()}")
    print(f"Test flowers - Present: {y_test_flowers.sum()}, Absent: {len(y_test_flowers) - y_test_flowers.sum()}")
    print(f"Test siliques - Present: {y_test_siliques.sum()}, Absent: {len(y_test_siliques) - y_test_siliques.sum()}")
    print(f"Test flowering status - FL: {y_test_flowering.sum()}, Non-FL: {len(y_test_flowering) - y_test_flowering.sum()}")
    
    
    # Model Training
    
    print("\n3. TRAINING CLASSIFIERS WITH HYPERPARAMETER OPTIMIZATION...")
    
    # Train flower classifier
    flower_results = train_calibrated_classifier(
        X_flower_train, y_flower_train,
        "flower",
        random_state=42
    )
    
    # Train silique classifier
    silique_results = train_calibrated_classifier(
        X_silique_train, y_silique_train,
        "silique",
        random_state=42
    )
    
    # Predictions
    
    print("\n4. MAKING PREDICTIONS WITH FIXED THRESHOLDS...")
    
    # Fixed thresholds (no optimization needed)
    FLOWER_THRESHOLD = 0.36
    SILIQUE_THRESHOLD = 0.385
    
    print(f"Using flower threshold: {FLOWER_THRESHOLD:.4f}")
    print(f"Using silique threshold: {SILIQUE_THRESHOLD:.4f}")
    
    # Get predictions
    flower_probs = flower_results['model'].predict_proba(X_test)[:, 1]
    flower_preds = (flower_probs >= FLOWER_THRESHOLD).astype(int)
    
    silique_probs = silique_results['model'].predict_proba(X_test)[:, 1]
    silique_preds = (silique_probs >= SILIQUE_THRESHOLD).astype(int)
    
    # Overall flowering classification using rule: Flowering = (flowers=1 AND siliques=0)
    overall_flowering_preds = ((flower_preds == 1) & (silique_preds == 0)).astype(int)
    
    print(f"\nPrediction Summary:")
    print(f"Flowers detected: {flower_preds.sum()}")
    print(f"Siliques detected: {silique_preds.sum()}")
    print(f"Overall flowering predicted: {overall_flowering_preds.sum()}")
    
    
    # Evaluation
    
    print("\n5. EVALUATION...")
    
    # Evaluate individual classifiers
    flower_metrics = evaluate_classifier(
        y_test_flowers, flower_preds, flower_probs,
        "flower", FLOWER_THRESHOLD
    )
    
    silique_metrics = evaluate_classifier(
        y_test_siliques, silique_preds, silique_probs,
        "silique", SILIQUE_THRESHOLD
    )
    
    # Evaluate overall flowering classification
    print(f"\n{'='*60}")
    print("OVERALL FLOWERING CLASSIFICATION EVALUATION")
    print(f"{'='*60}")
    print(f"Rule: Flowering = (flowers=1 AND siliques=0)")
    
    # Overall flowering classification report
    print(f"\nOverall Flowering Classification Report:")
    print(classification_report(
        y_test_flowering, overall_flowering_preds,
        target_names=['Non-Flowering', 'Flowering'],
        zero_division=0
    ))
    
    # Overall flowering confusion matrix
    cm_overall = confusion_matrix(y_test_flowering, overall_flowering_preds)
    print(f"\nOverall Flowering Confusion Matrix:")
    print(f"                        Predicted")
    print(f"                    Non-FL    Flowering")
    print(f"Actual  Non-FL        {cm_overall[0,0]:4d}        {cm_overall[0,1]:4d}")
    print(f"        Flowering     {cm_overall[1,0]:4d}        {cm_overall[1,1]:4d}")
    print(f"        --------------------------")
    print(f"        Total:        {cm_overall.sum():4d}        (100%)")
    
    # Overall flowering metrics
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
        y_test_flowering, overall_flowering_preds, average='binary', zero_division=0
    )
    overall_accuracy = accuracy_score(y_test_flowering, overall_flowering_preds)
    
    print(f"\nOverall Flowering Key Metrics:")
    print(f"  Accuracy: {overall_accuracy:.4f}")
    print(f"  Precision (Flowering): {overall_precision:.4f}")
    print(f"  Recall (Flowering): {overall_recall:.4f}")
    print(f"  F1-Score (Flowering): {overall_f1:.4f}")
    
    # Detailed breakdown
    print(f"\n{'='*60}")
    print("DUAL CLASSIFICATION BREAKDOWN")
    print(f"{'='*60}")
    
    flowers_only = np.sum((flower_preds == 1) & (silique_preds == 0))
    siliques_only = np.sum((flower_preds == 0) & (silique_preds == 1))
    both_present = np.sum((flower_preds == 1) & (silique_preds == 1))
    neither_present = np.sum((flower_preds == 0) & (silique_preds == 0))
    
    print(f"Images with flowers only (FLOWERING): {flowers_only}")
    print(f"Images with siliques only (NON-FLOWERING): {siliques_only}")
    print(f"Images with both flowers and siliques (NON-FLOWERING): {both_present}")
    print(f"Images with neither flowers nor siliques (NON-FLOWERING): {neither_present}")
    
   
    # Saving models 

    print("\n6. SAVING MODELS...")
    
    # Save the trained models
    joblib.dump(flower_results['model'], '../models/trained_flower_classifier.joblib')
    joblib.dump(silique_results['model'], '../models/trained_silique_classifier.joblib')
    
    # Save the model metadata
    model_info = {
        'flower_threshold': FLOWER_THRESHOLD,
        'silique_threshold': SILIQUE_THRESHOLD,
        'flower_params': flower_results['best_params'],
        'silique_params': silique_results['best_params'],
        'flower_cv_score': flower_results['cv_score'],
        'silique_cv_score': silique_results['cv_score'],
        'flower_metrics': flower_metrics,
        'silique_metrics': silique_metrics,
        'overall_accuracy': overall_accuracy,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_f1': overall_f1
    }
    
    with open('../models/trained_dual_classifier_info.pkl', 'wb') as f:
        pickle.dump(model_info, f)
    
    print("Models saved:")
    print("  - ../models/trained_flower_classifier.joblib")
    print("  - ../models/trained_silique_classifier.joblib") 
    print("  - ../models/trained_dual_classifier_info.pkl")
    
    # Final Summary
    
    print(f"\n{'='*80}")
    print("TRAINING AND TESTING COMPLETE - FINAL SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nHyperparameter Optimization Results:")
    print(f"  Flower Classifier:")
    print(f"    Best CV Score: {flower_results['cv_score']:.4f}")
    print(f"    Best Params: {flower_results['best_params']}")
    print(f"  Silique Classifier:")
    print(f"    Best CV Score: {silique_results['cv_score']:.4f}")
    print(f"    Best Params: {silique_results['best_params']}")
    
    print(f"\nIndividual Classifier Performance:")
    print(f"  Flower Classifier:")
    print(f"    Accuracy: {flower_metrics['accuracy']:.4f}")
    print(f"    Precision: {flower_metrics['precision']:.4f}")
    print(f"    Recall: {flower_metrics['recall']:.4f}")
    print(f"    F1-Score: {flower_metrics['f1']:.4f}")
    print(f"    Threshold: {FLOWER_THRESHOLD:.4f}")
    
    print(f"\n  Silique Classifier:")
    print(f"    Accuracy: {silique_metrics['accuracy']:.4f}")
    print(f"    Precision: {silique_metrics['precision']:.4f}")
    print(f"    Recall: {silique_metrics['recall']:.4f}")
    print(f"    F1-Score: {silique_metrics['f1']:.4f}")
    print(f"    Threshold: {SILIQUE_THRESHOLD:.4f}")
    
    print(f"\nOverall Flowering Classification:")
    print(f"    Rule: Flowering = (flowers=1 AND siliques=0)")
    print(f"    Accuracy: {overall_accuracy:.4f}")
    print(f"    Precision: {overall_precision:.4f}")
    print(f"    Recall: {overall_recall:.4f}")
    print(f"    F1-Score: {overall_f1:.4f}")
    
    print(f"\nDual classifier training and testing completed successfully!")
    
    return model_info

if __name__ == "__main__":
    results = main()