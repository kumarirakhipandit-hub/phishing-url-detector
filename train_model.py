import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("processed_dataset.csv")

# Separate features and label
X = data.drop("label", axis=1)
y = data["label"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model trained successfully!")
print("Accuracy:", accuracy)

# Save model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved as model.pkl")