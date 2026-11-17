import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

from src.preprocessing import clean_text

# Dataset path
DATA_PATH = Path("3. Hate Speech Detection.csv")

# Load dataset
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

# Check columns
print("Columns:", df.columns.tolist())

# Select required columns
df = df[["Text", "Lebel"]].dropna()

# Clean text
df["Text"] = df["Text"].apply(clean_text)
df["Lebel"] = df["Lebel"].astype(str).str.strip()

# Remove empty and duplicate data
df = df[df["Text"] != ""].drop_duplicates()

X = df["Text"]
y = df["Lebel"]

print("Dataset size:", len(df))
print("Label distribution:")
print(y.value_counts())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create model
model = Pipeline([
    ("features", FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50000
        )),
        ("char", TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            max_features=50000
        ))
    ])),
    ("classifier", LinearSVC())
])

# Train
print("Training started...")
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(
    y_test, predictions,
    zero_division=0
))

# Save model
joblib.dump(model, "model.joblib")

print("Model saved successfully!")
