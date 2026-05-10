import os
import librosa
import numpy as np

# Dataset path
DATASET_PATH = "../../data"

# Store features and labels
features = []
labels = []

# Function to extract MFCC features
def extract_features(file_path):

    # Load audio
    audio, sample_rate = librosa.load(file_path, sr=22050)

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    # Convert to fixed-size vector
    mfccs_processed = np.mean(mfccs.T, axis=0)

    return mfccs_processed

# Loop through dataset folders
for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        # Check audio files
        if file.endswith(".wav"):

            file_path = os.path.join(root, file)

            try:
                # Extract emotion label
                emotion = file.split("_")[-1].split(".")[0]

                # Extract features
                feature = extract_features(file_path)

                features.append(feature)
                labels.append(emotion)

                print(f"Processed: {file}")

            except Exception as e:
                print(f"Error processing {file}")
                print(e)

# Convert into arrays
X = np.array(features)
y = np.array(labels)

print("\nFeature Shape:", X.shape)
print("Labels Shape:", y.shape)

# Save processed data
np.save("X_features.npy", X)
np.save("y_labels.npy", y)

print("\nPreprocessing completed successfully!")