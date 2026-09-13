import pandas as pd
import re
from urllib.parse import urlparse


def extract_features(url):
    url = str(url)

    parsed_url = urlparse(url)

    features = {
        "url_length": len(url),
        "has_https": 1 if parsed_url.scheme == "https" else 0,
        "has_at_symbol": 1 if "@" in url else 0,
        "has_dash": 1 if "-" in url else 0,
        "has_dot": 1 if "." in url else 0,
        "has_ip": 1 if re.search(r"\b\d{1,3}(\.\d{1,3}){3}\b", url) else 0,
        "has_query": 1 if "?" in url else 0,
        "has_fragment": 1 if "#" in url else 0
    }

    return features


# Read the dataset
df = pd.read_csv("dataset.csv")

# Extract features from every URL
feature_data = df["url"].apply(extract_features)

# Convert features into a DataFrame
features_df = pd.DataFrame(feature_data.tolist())

# Add the original label
features_df["label"] = df["label"]

# Save the processed dataset
features_df.to_csv("processed_dataset.csv", index=False)

print("Feature extraction completed successfully!")
print(features_df)