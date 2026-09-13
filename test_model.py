import pandas as pd
import joblib
from feature_extraction import extract_features

model = joblib.load("model.pkl")

url = input("Enter a URL: ")

features = extract_features(url)

features_df = pd.DataFrame([features])
prediction = model.predict(features_df)[0]

if prediction == 1:
    print("⚠️ Phishing URL")
else:
    print("✅ Legitimate URL")