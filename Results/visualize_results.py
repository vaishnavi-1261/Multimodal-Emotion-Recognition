import matplotlib.pyplot as plt

# Model names
models = [
    "Speech Model",
    "Text Model",
    "Fusion Model"
]

# Example accuracies
accuracies = [
    99.9,
    90,
    98
]

# Create bar graph
plt.figure(figsize=(8, 5))

plt.bar(models, accuracies)

plt.xlabel("Models")
plt.ylabel("Accuracy (%)")

plt.title("Emotion Recognition Model Comparison")

# Save graph
plt.savefig("model_comparison.png")

# Show graph
plt.show()

print("Visualization created successfully!")