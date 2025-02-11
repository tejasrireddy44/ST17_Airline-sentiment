import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# ✅ Add a text preprocessing function
def preprocess_text(text):
    """
    Preprocesses input text by lowercasing and stripping extra spaces.
    """
    if not isinstance(text, str):  # Ensure input is a string
        return ""
    return text.lower().strip()

def train_model():
    # Define file paths
    input_file = "c:/Users/TEJASRI REDDY.V/Desktop/ST17_Airline _sentiment/data/cleaned_airline_tweets.csv"
    model_dir = "c:/Users/TEJASRI REDDY.V/Desktop/ST17_Airline _sentiment/models"
    model_file = os.path.join(model_dir, "sentiment_model.pkl")
    vectorizer_file = os.path.join(model_dir, "tfidf_vectorizer.pkl")

    # Ensure the models directory exists
    os.makedirs(model_dir, exist_ok=True)  

    # Load data
    df = pd.read_csv(input_file)

    # Ensure 'cleaned_text' and 'airline_sentiment' columns exist
    if 'cleaned_text' not in df.columns or 'airline_sentiment' not in df.columns:
        raise KeyError("Missing required columns: 'cleaned_text' or 'airline_sentiment' in the dataset.")

    # ✅ Preprocess text before vectorization
    df['cleaned_text'] = df['cleaned_text'].apply(preprocess_text)

    # Feature extraction & splitting data
    X = df['cleaned_text']
    y = df['airline_sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build text classification model
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_transformed = vectorizer.fit_transform(X_train)  # Ensure text is transformed correctly
    X_test_transformed = vectorizer.transform(X_test)

    # Train classifier
    model = MultinomialNB()
    model.fit(X_train_transformed, y_train)

    # Save trained model & vectorizer
    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    with open(vectorizer_file, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Model saved at: {model_file}")
    print(f"Vectorizer saved at: {vectorizer_file}")

if __name__ == "__main__":
    train_model()
