import streamlit as st
import joblib
from streamlit_mic_recorder import mic_recorder

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="Emotion Recognition AI",
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
    font-size: 52px;
    font-weight: bold;
    color: white;
    margin-top: 20px;
}

.subtitle {
    text-align: center;
    color: #BBBBBB;
    font-size: 20px;
    margin-bottom: 40px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    margin-top: 25px;
    background-color: #1E1E1E;
    border: 1px solid #333333;
    text-align: center;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 60px;
}

.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 50px;
    font-size: 20px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# LOAD NLP MODEL
# ------------------------------------------------

text_model = joblib.load(
    "models/text_pipeline/text_emotion_model.pkl"
)

vectorizer = joblib.load(
    "models/text_pipeline/tfidf_vectorizer.pkl"
)

# ------------------------------------------------
# HEADER
# ------------------------------------------------

st.markdown(
    '<div class="title">🎭 Emotion Recognition AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Real-Time Mood Detection using NLP & AI</div>',
    unsafe_allow_html=True
)

# ------------------------------------------------
# AUDIO UI (FOR INTERACTION PURPOSE)
# ------------------------------------------------

st.subheader("🎤 Voice Interaction")

st.write("Record your voice or upload audio for interaction.")

recorded_audio = mic_recorder(
    start_prompt="🎙 Start Recording",
    stop_prompt="⏹ Stop Recording",
    just_once=True
)

uploaded_audio = st.file_uploader(
    "📁 Upload WAV Audio File",
    type=["wav"]
)

# ------------------------------------------------
# TEXT INPUT
# ------------------------------------------------

st.subheader("✍ Enter Your Message")

text_input = st.text_area(
    "",
    placeholder="Type how you feel..."
)

# ------------------------------------------------
# EMOTION PREDICTION
# ------------------------------------------------

if st.button("🚀 Detect Emotion"):

    if text_input != "":

        text_vectorized = vectorizer.transform(
            [text_input]
        )

        predicted_emotion = text_model.predict(
            text_vectorized
        )[0]

        # Emotion emoji mapping
        emoji_map = {
            "happy": "😊",
            "sad": "😢",
            "anger": "😡",
            "angry": "😡",
            "fear": "😨",
            "surprise": "😲",
            "love": "❤️",
            "neutral": "😐"
        }

        emoji = emoji_map.get(
            predicted_emotion.lower(),
            "🎭"
        )

        # Smart emotional explanation
        emotion_message = {
            "happy": "You seem cheerful and positive.",
            "sad": "You may be feeling emotionally low.",
            "anger": "You seem frustrated or angry.",
            "angry": "You seem frustrated or angry.",
            "fear": "You may be feeling anxious or fearful.",
            "surprise": "You seem surprised or shocked.",
            "love": "You seem affectionate and warm.",
            "neutral": "Your emotional state appears calm."
        }

        explanation = emotion_message.get(
            predicted_emotion.lower(),
            "Emotion detected successfully."
        )

        st.markdown(f"""
        <div class="result-box">

        <h1>{emoji}</h1>

        <h2 style="color:#00FFAA;">
        Detected Emotion:
        {predicted_emotion.upper()}
        </h2>

        <p style="font-size:18px; color:#DDDDDD;">
        {explanation}
        </p>

        </div>
        """, unsafe_allow_html=True)

    else:
        st.warning("⚠ Please enter some text.")

# ------------------------------------------------
# FOOTER
# ------------------------------------------------

st.markdown(
    '<div class="footer">Built with ❤️ using Streamlit & NLP</div>',
    unsafe_allow_html=True
)