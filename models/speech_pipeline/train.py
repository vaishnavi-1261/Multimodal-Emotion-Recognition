import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load features and labels
X = np.load("X_features.npy")
y = np.load("y_labels.npy")

print("Features Shape:", X.shape)
print("Labels Shape:", y.shape)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Convert labels to categorical
y_categorical = to_categorical(y_encoded)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.2,
    random_state=42
)

# Build model
model = Sequential()

model.add(Dense(256, activation='relu', input_shape=(40,)))
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))

# Output layer
model.add(Dense(y_categorical.shape[1], activation='softmax'))

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
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", accuracy)

# Save model
model.save("speech_emotion_model.h5")

print("\nModel saved successfully!")