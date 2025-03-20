import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Step 1: Set the dataset path
dataset_path = os.path.join(os.getcwd(), "train.tsv")

# Step 2: Load the LIAR Dataset
df = pd.read_csv(dataset_path, sep='\t', header=None, names=[
    "id", "label", "statement", "subject", "speaker", "job_title", "state", "party", "barely_true_counts",
    "false_counts", "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context"
])

# Step 3: Keep only necessary columns (Statement & Label)
df = df[["statement", "label"]]

# Step 4: Convert labels to binary (Fake = 1, Real = 0)
label_mapping = {
    "true": 0, "mostly-true": 0, "half-true": 1,
    "barely-true": 1, "false": 1, "pants-fire": 1
}
df['label'] = df['label'].map(label_mapping)

# Step 5: Convert Text to TF-IDF Features
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X = vectorizer.fit_transform(df['statement'])
y = df['label']

# Step 6: Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 7: Train XGBoost Model with Hyperparameters
model = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Step 8: Save the Model and TF-IDF Vectorizer
joblib.dump(model, "fake_news_xgboost.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Step 9: Evaluate Model Performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Step 10: Print Performance Metrics
print(f"\n📌 Fake News Detection Model Accuracy: {accuracy:.2f}")
print("\n📌 Classification Report:\n", classification_report(y_test, y_pred))
print("\n📌 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 11: Feature Importance (Top words influencing Fake News Detection)
feature_names = vectorizer.get_feature_names_out()
importances = model.feature_importances_
top_features = np.argsort(importances)[-20:]  # Top 20 most important words

print("\n📌 Top Words Influencing Fake News Detection:")
for i in top_features:
    print(feature_names[i], ":", round(importances[i], 4))
