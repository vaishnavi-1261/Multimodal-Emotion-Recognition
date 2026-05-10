import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

from tensorflow.keras.utils import to_categorical

# Load dataset
df = pd.read_csv("better_text_dataset.csv")
# Text and labels
texts = df["text"].values
labels = df["emotion"].values

# Encode labels
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)

# Convert labels to categorical
labels_categorical = to_categorical(labels_encoded)

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(texts)

# Padding
X = pad_sequences(sequences, maxlen=10)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels_categorical,
    test_size=0.2,
    random_state=42
)

# Vocabulary size
vocab_size = len(tokenizer.word_index) + 1

# Build model
model = Sequential()

model.add(Embedding(
    input_dim=vocab_size,
    output_dim=64,
    input_length=10
))

model.add(LSTM(64))

model.add(Dense(64, activation='relu'))

model.add(Dense(
    labels_categorical.shape[1],
    activation='softmax'
))

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)

print("\nText Model Accuracy:", accuracy)

# Save model
model.save("text_emotion_model.h5")

print("\nText model saved successfully!")