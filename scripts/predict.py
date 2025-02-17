import os
import sys
import pickle
import pandas as pd

# Get the project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add 'scripts' directory to Python path
sys.path.append(os.path.join(BASE_DIR, "scripts"))

# Import preprocess function
from preprocess import clean_dataset  # Import from preprocess.py

# Define paths for model and vectorizer
MODEL_FILE = os.path.join(BASE_DIR, "models", "sentiment_model.pkl")
VECTORIZER_FILE = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")

# Ensure model and vectorizer exist
if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):
    raise FileNotFoundError(" Model or vectorizer file not found. Train the model first.")

# Load trained sentiment model
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

# Load TF-IDF vectorizer
with open(VECTORIZER_FILE, "rb") as f:
    vectorizer = pickle.load(f)

# Print vectorizer vocabulary size for debugging
print(f"Vectorizer Vocabulary Size: {len(vectorizer.get_feature_names_out())}")

def predict_sentiment(text):
    """
    Predict the sentiment of a given text input.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return "Invalid input"

    # Convert input to lowercase
    cleaned_text = text.lower()
    print(f"Cleaned Text: {cleaned_text}")

    # Ensure input is a list when transforming
    text_vectorized = vectorizer.transform([cleaned_text])  # Correctly vectorize input
    print(f" Transformed Text Shape: {text_vectorized.shape}")  # Debugging

    # Make sure vectorization worked
    if text_vectorized.shape[1] == 0:
        return "Error: No matching words in vocabulary"

    #  **Final Fix: Ensure matrix is passed correctly to the model**
    try:
        prediction = model.predict(text_vectorized)  # No `[0]` at this step
        print(f" Model Output: {prediction}")  # Debugging output
    except Exception as e:
        print(f" Error in prediction: {e}")
        return "Prediction failed"

    return prediction[0]  # Extract first value from NumPy array

if __name__ == "__main__":
    print("Enter a tweet: ", end="")
    test_text = input().strip()

    sentiment = predict_sentiment(test_text)

    print(f"Predicted Sentiment: {sentiment}")

    print(f"Predicted Sentiment: {sentiment}")

