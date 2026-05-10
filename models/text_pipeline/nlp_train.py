import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import joblib

# Load dataset
df = pd.read_csv("processed_text_dataset.csv")

# Text and labels
X = df["text"]
y = df["emotion"]

# Convert text into TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)

X_vectorized = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nText Model Accuracy:", accuracy)

# Classification report
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

# Confusion matrix
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, predictions))

# Save model
joblib.dump(model, "text_emotion_model.pkl")

# Save vectorizer
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nNLP model saved successfully!")