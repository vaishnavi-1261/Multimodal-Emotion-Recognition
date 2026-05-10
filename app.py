import streamlit as st
import numpy as np
import joblib
import tempfile
from sklearn.preprocessing import LabelEncoder
from streamlit_mic_recorder import mic_recorder

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="Multimodal Emotion Recognition",
    page_icon="🎭",
    layout="centered"
)

# ------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: white;
}

.subtitle {
    text-align: center;
    color: #BBBBBB;
    font-size: 20px;
    margin-bottom: 30px;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
    background-color: #1E1E1E;
    border: 1px solid #444;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 50px;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# LOAD MODELS
# ------------------------------------------------

speech_model = load_model(
    "models/speech_pipeline/speech_emotion_model.h5"
)

speech_labels = np.load(
    "models/speech_pipeline/y_labels.npy"
)

speech_encoder = LabelEncoder()
speech_encoder.fit(speech_labels)

text_model = joblib.load(
    "models/text_pipeline/text_emotion_model.pkl"
)

vectorizer = joblib.load(
    "models/text_pipeline/tfidf_vectorizer.pkl"
)

# ------------------------------------------------
# FEATURE EXTRACTION
# ------------------------------------------------

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

# ------------------------------------------------
# HEADER
# ------------------------------------------------

st.markdown(
    '<div class="title">🎭 Multimodal Emotion Recognition</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Real-Time Speech + Text Emotion Detection using AI</div>',
    unsafe_allow_html=True
)

# ------------------------------------------------
# AUDIO INPUT OPTIONS
# ------------------------------------------------

st.subheader("🎤 Voice Input")

st.write("You can either record live audio or upload a WAV file.")

# Live recording
recorded_audio = mic_recorder(
    start_prompt="🎙 Start Recording",
    stop_prompt="⏹ Stop Recording",
    just_once=True
)

# Upload audio option
uploaded_audio = st.file_uploader(
    "📁 Upload WAV Audio File",
    type=["wav"]
)

# Final audio selection
final_audio = None

if recorded_audio:
    final_audio = recorded_audio["bytes"]

elif uploaded_audio is not None:
    final_audio = uploaded_audio.read()

# ------------------------------------------------
# TEXT INPUT
# ------------------------------------------------

text_input = st.text_input(
    "✍ Enter Text"
)

# ------------------------------------------------
# PREDICT BUTTON
# ------------------------------------------------

if st.button("🚀 Predict Emotion"):

    if final_audio and text_input != "":

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as tmp_file:

            tmp_file.write(final_audio)

            temp_audio_path = tmp_file.name

        # ----------------------------------------
        # SPEECH PREDICTION
        # ----------------------------------------

        speech_features = extract_features(temp_audio_path)

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

        # ----------------------------------------
        # TEXT PREDICTION
        # ----------------------------------------

        text_vectorized = vectorizer.transform(
            [text_input]
        )

        text_emotion = text_model.predict(
            text_vectorized
        )[0]

        # ----------------------------------------
        # ADVANCED FUSION LOGIC
        # ----------------------------------------

        # Speech captures hidden emotions better
        # especially when user speaks normally but emotionally.

        negative_emotions = ["sad", "sadness", "angry", "anger", "fear"]

        if speech_emotion == text_emotion:
            final_emotion = speech_emotion

        elif speech_emotion.lower() in negative_emotions:
            # Prioritize speech for hidden emotions
            final_emotion = speech_emotion

        else:
            final_emotion = text_emotion

        # ----------------------------------------
        # RESULTS
        # ----------------------------------------

        st.markdown(f"""
        <div class="result-box">

        <h3 style="color:#00FFAA;">
        🎤 Speech Emotion: {speech_emotion}
        </h3>

        <h3 style="color:#00BFFF;">
        📝 Text Emotion: {text_emotion}
        </h3>

        <h3 style="color:#FFD700;">
        🎯 Final Emotion: {final_emotion}
        </h3>

        </div>
        """, unsafe_allow_html=True)

    else:
        st.warning("⚠ Please provide audio input and enter text.")

# ------------------------------------------------
# FOOTER
# ------------------------------------------------

st.markdown(
    '<div class="footer">Built with ❤️ using Streamlit, TensorFlow & NLP</div>',
    unsafe_allow_html=True
)