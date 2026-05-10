import joblib

# Load model
model = joblib.load("text_emotion_model.pkl")

# Load vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Example text
text = ["I am very angry"]
# Convert text into TF-IDF
text_vectorized = vectorizer.transform(text)

# Predict emotion
prediction = model.predict(text_vectorized)

print("\nPredicted Emotion:", prediction[0])