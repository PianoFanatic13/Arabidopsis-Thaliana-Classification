import os
import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Load embeddings
with open("train_embeddings.pkl", "rb") as f:
    train_embeddings = pickle.load(f)
with open("test_embeddings.pkl", "rb") as f:
    test_embeddings = pickle.load(f)

# Load labels
train_df = pd.read_csv("Training Set.csv", dtype=str)
test_df = pd.read_csv("Testing Set.csv", dtype=str)

# Prepare X and y
X_train = np.array([train_embeddings[os.path.join("images", f"image_{img_num}.jpg")] for img_num in train_df["Image #"]])
y_train = train_df["Classification"].values

X_test = np.array([test_embeddings[os.path.join("images", f"image_{img_num}.jpg")] for img_num in test_df["Image #"]])
y_test = test_df["Classification"].values

# Encode labels for XGBoost
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# SVM
clf = SVC(kernel="linear")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("SVM Results:")
print(classification_report(y_test, y_pred))

# Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Random Forest Results:")
print(classification_report(y_test, y_pred))

# XGBoost Classifier
clf = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=True, eval_metric='logloss')
clf.fit(X_train, y_train_encoded)
y_pred_encoded = clf.predict(X_test)
y_pred = le.inverse_transform(y_pred_encoded)
print("XGBoost Results:")
print(classification_report(y_test, y_pred))

from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestClassifier

search_space = {
    'n_estimators': (50, 300),
    'max_depth': (2, 20),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 10)
}

opt = BayesSearchCV(
    RandomForestClassifier(random_state=42),
    search_space,
    n_iter=32,  # Number of parameter settings sampled
    cv=3,      # Cross-validation folds
    scoring='accuracy',
    n_jobs=-1
)

# After running BayesSearchCV
best_params = dict([('max_depth', 9), ('min_samples_leaf', 1), ('min_samples_split', 7), ('n_estimators', 202)])

# Create and train a new Random Forest with tuned hyperparameters
clf = RandomForestClassifier(**best_params, random_state=42)
clf.fit(X_train, y_train)

# Evaluate on test set
y_pred = clf.predict(X_test)
print("Random Forest (Tuned) Results:")
print(classification_report(y_test, y_pred))