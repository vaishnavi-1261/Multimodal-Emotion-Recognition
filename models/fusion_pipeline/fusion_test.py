import librosa
import numpy as np
import joblib

from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# LOAD SPEECH MODEL
# -----------------------------

speech_model = load_model(
    "../speech_pipeline/speech_emotion_model.h5"
)

speech_labels = np.load(
    "../speech_pipeline/y_labels.npy"
)

speech_encoder = LabelEncoder()
speech_encoder.fit(speech_labels)

# -----------------------------
# LOAD TEXT MODEL
# -----------------------------

text_model = joblib.load(
    "../text_pipeline/text_emotion_model.pkl"
)

vectorizer = joblib.load(
    "../text_pipeline/tfidf_vectorizer.pkl"
)

# -----------------------------
# SPEECH FEATURE EXTRACTION
# -----------------------------

def extract_features(file_path):

    audio, sample_rate = librosa.load(
        file_path,
        sr=22050
    )

    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    mfccs_processed = np.mean(
        mfccs.T,
        axis=0
    )

    return mfccs_processed

# -----------------------------
# INPUTS
# -----------------------------

audio_file = "../../data/TESS Toronto emotional speech set data/OAF_angry/OAF_back_angry.wav"

text_input = ["I am very angry today"]

# -----------------------------
# SPEECH PREDICTION
# -----------------------------

speech_features = extract_features(audio_file)

speech_features = np.expand_dims(
    speech_features,
    axis=0
)

speech_prediction = speech_model.predict(
    speech_features
)

speech_emotion = speech_encoder.inverse_transform(
    [np.argmax(speech_prediction)]
)[0]

# -----------------------------
# TEXT PREDICTION
# -----------------------------

text_vectorized = vectorizer.transform(
    text_input
)

text_emotion = text_model.predict(
    text_vectorized
)[0]

# -----------------------------
# FUSION LOGIC
# -----------------------------

if speech_emotion == text_emotion:
    final_emotion = speech_emotion

else:
    # Prioritize speech emotion
    final_emotion = speech_emotion

# -----------------------------
# OUTPUT
# -----------------------------

print("\nSpeech Emotion :", speech_emotion)
print("Text Emotion   :", text_emotion)
print("Final Emotion  :", final_emotion)