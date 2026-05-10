import pandas as pd

# Function to load dataset
def load_data(file_path):

    texts = []
    labels = []

    with open(file_path, 'r', encoding='utf-8') as file:

        for line in file:

            parts = line.strip().split(';')

            if len(parts) == 2:
                text, emotion = parts

                texts.append(text)
                labels.append(emotion)

    return pd.DataFrame({
        "text": texts,
        "emotion": labels
    })

# Load train dataset
df = load_data("../../data/text_dataset/train.txt")

print(df.head())

# Save processed CSV
df.to_csv("processed_text_dataset.csv", index=False)

print("\nNLP preprocessing completed successfully!")