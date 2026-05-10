import pandas as pd

data = {
    "text": [
        # Happy
        "I am very happy today",
        "This is amazing",
        "I got my dream job",
        "I feel wonderful",
        "Life is beautiful",
        "I am smiling a lot",

        # Sad
        "I feel so sad",
        "I am crying badly",
        "I lost my friend",
        "I feel depressed",
        "Everything hurts",
        "I feel lonely",

        # Angry
        "Why are you shouting",
        "I am extremely angry",
        "This makes me furious",
        "I hate this situation",
        "Stop irritating me",
        "I am very upset",

        # Fear
        "I am scared",
        "This is frightening",
        "I am terrified",
        "I fear the dark",
        "Something dangerous is coming",
        "I feel nervous",

        # Surprise
        "Wow this is surprising",
        "I cannot believe this",
        "This shocked me",
        "What an unexpected moment",
        "This is unbelievable",
        "I am astonished",

        # Neutral
        "Everything is normal",
        "I am going to school",
        "Today is Monday",
        "I am sitting quietly",
        "The weather is fine",
        "I am reading a book"
    ],

    "emotion": [
        "happy","happy","happy","happy","happy","happy",
        "sad","sad","sad","sad","sad","sad",
        "angry","angry","angry","angry","angry","angry",
        "fear","fear","fear","fear","fear","fear",
        "surprise","surprise","surprise","surprise","surprise","surprise",
        "neutral","neutral","neutral","neutral","neutral","neutral"
    ]
}

df = pd.DataFrame(data)

df.to_csv("better_text_dataset.csv", index=False)

print(df)

print("\nLarge emotional text dataset created successfully!")