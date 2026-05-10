import os
import pandas as pd

# Dataset path
DATASET_PATH = "../../data/TESS Toronto emotional speech set data"

# Store data
texts = []
labels = []

# Loop through dataset
for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        if file.endswith(".wav"):

            try:
                # Example:
                # OAF_back_angry.wav

                parts = file.replace(".wav", "").split("_")

                # Extract text
                text = parts[1]

                # Extract emotion
                emotion = parts[2]

                texts.append(text)
                labels.append(emotion)

            except Exception as e:
                print("Error:", file)
                print(e)

# Create dataframe
df = pd.DataFrame({
    "text": texts,
    "emotion": labels
})

# Save CSV
df.to_csv("text_emotion_dataset.csv", index=False)

print(df.head())

print("\nDataset created successfully!")