import librosa
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load model
model = load_model("speech_emotion_model.h5")

# Emotion labels
labels = np.load("y_labels.npy")

# Encode labels
encoder = LabelEncoder()
encoder.fit(labels)

# Function to extract MFCC features
def extract_features(file_path):

    audio, sample_rate = librosa.load(file_path, sr=22050)

    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    mfccs_processed = np.mean(mfccs.T, axis=0)

    return mfccs_processed

# Test audio file
test_file = "../../data/TESS Toronto emotional speech set data/OAF_angry/OAF_back_angry.wav"

# Extract features
features = extract_features(test_file)

# Reshape for prediction
features = np.expand_dims(features, axis=0)

# Predict
prediction = model.predict(features)

# Get predicted label
predicted_label = encoder.inverse_transform(
    [np.argmax(prediction)]
)

print("\nPredicted Emotion:", predicted_label[0])