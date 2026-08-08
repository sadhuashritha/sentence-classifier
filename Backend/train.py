import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ==========================================
# Load Dataset
# ==========================================

data_path = os.path.join("..", "dataset", "train.csv")

df = pd.read_csv(data_path)


# ==========================================
# Create Binary Label
# ==========================================

label_columns = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

df["label"] = df[label_columns].max(axis=1)


# ==========================================
# Features and Labels
# ==========================================

X = df["comment_text"]
y = df["label"]


# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))


# ==========================================
# TF-IDF Vectorization
# ==========================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print("\nText converted into vectors successfully.")


# ==========================================
# Train Model
# ==========================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("Model trained successfully.")


# ==========================================
# Prediction
# ==========================================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nAccuracy : {accuracy * 100:.2f}%")

print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, predictions))


# ==========================================
# Save Model
# ==========================================

os.makedirs("saved_model", exist_ok=True)

joblib.dump(model, "saved_model/model.pkl")
joblib.dump(vectorizer, "saved_model/vectorizer.pkl")

print("\nModel saved successfully.")
print("saved_model/model.pkl")
print("saved_model/vectorizer.pkl")