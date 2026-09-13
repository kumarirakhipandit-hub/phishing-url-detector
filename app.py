import streamlit as st
import pandas as pd
import joblib

from feature_extraction import extract_features


# Load trained model
model = joblib.load("model.pkl")


# Page configuration
st.set_page_config(
    page_title="AI Phishing URL Detector",
    page_icon="🔐",
    layout="centered"
)


# Title
st.title("🔐 AI-Based Phishing URL Detection System")

st.write(
    "This application uses Machine Learning to classify a URL "
    "as Legitimate or potentially Phishing."
)

st.divider()


# URL input
url = st.text_input(
    "🌐 Enter Website URL",
    placeholder="Example: https://www.google.com"
)


# Check button
if st.button("🔍 Check URL", use_container_width=True):

    if not url.strip():
        st.warning("⚠️ Please enter a URL first.")

    else:
        # Extract features
        features = extract_features(url)

        # Convert features into DataFrame
        features_df = pd.DataFrame([features])

        # Prediction
        prediction = model.predict(features_df)[0]

        st.divider()

        # Display result
        if prediction == 1:
            st.error("⚠️ PHISHING URL DETECTED")
            st.write(
                "This URL has characteristics associated with "
                "phishing websites."
            )

        else:
            st.success("✅ LEGITIMATE URL")
            st.write(
                "This URL appears to be legitimate based on "
                "the trained machine learning model."
            )

        # Feature information
        with st.expander("📊 View Extracted URL Features"):
            st.dataframe(features_df)