import pandas as pd
import joblib
import re
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Dataset path
DATA_PATH = Path("3. Hate Speech Detection.csv")

# Load dataset
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

# Check columns
print("Dataset columns:", df.columns.tolist())

# Dataset has Text and Lebel columns
df = df[["Text", "Lebel"]].dropna()

# Convert text and labels
df["Text"] = df["Text"].astype(str).str.strip()
df["Lebel"] = df["Lebel"].astype(str).str.strip()

# Remove empty text and duplicate rows
df = df[df["Text"] != ""].drop_duplicates()

X = df["Text"]
y = df["Lebel"]

print("Total samples:", len(df))
print("Labels:", y.value_counts())

# Check dataset
if y.nunique() < 2 or y.value_counts().min() < 2:
    raise ValueError("Dataset needs at least 2 samples per class.")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Build model
model = Pipeline([
    ("features", FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50000,
            sublinear_tf=True
        )),
        ("char", TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            max_features=50000,
            sublinear_tf=True
        ))
    ])),
    ("classifier", LinearSVC())
])

# Train model
print("\nTraining started...")
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Save model
joblib.dump(model, "model.joblib")

print("\nModel saved successfully!")
