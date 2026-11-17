import streamlit as st
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Bangla Hate Speech Detection",
    page_icon="🛡️",
    layout="centered"
)

# Title
st.title("🛡️ Bangla Hate Speech Detection")
st.write(
    "Enter a Bangla sentence to detect "
    "whether it contains hate speech."
)

# Load trained model
MODEL_PATH = Path("model.joblib")

if not MODEL_PATH.exists():
    st.error(
        "Model not found! Please run train.py first "
        "to train and save the model."
    )
    st.stop()

model = joblib.load(MODEL_PATH)

# User input
text = st.text_area(
    "Enter Bangla Text:",
    placeholder="এখানে বাংলা বাক্য লিখুন..."
)

# Prediction
if st.button("Detect Hate Speech"):
    if text.strip():
        prediction = model.predict([text])[0]

        st.subheader("Prediction Result")

        label = str(prediction).strip().lower()

        if label in ["hate", "1", "1.0", "hate speech"]:
            st.error("⚠️ Hate Speech Detected")
        elif label in ["non-hate", "non hate", "0", "0.0"]:
            st.success("✅ Non-Hate Speech")
        else:
            st.info(f"Predicted Label: {prediction}")
    else:
        st.warning("Please enter some text.")

st.divider()

st.caption("Bangla NLP Project | Hate Speech Detection")
