import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources if not available
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

def clean_text(text):
    """Cleans and preprocesses text."""
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove mentions, hashtags, and URLs
    text = re.sub(r"@\w+|#\w+|http\S+", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)

def clean_dataset(input_file="c:/Users/TEJASRI REDDY.V/Desktop/ST17_Airline _sentiment/data/Tweets.csv"):
    # Load dataset
    df = pd.read_csv(input_file)

    # Ensure required columns exist
    if "text" not in df.columns or "airline_sentiment" not in df.columns:
        raise KeyError("Dataset must contain 'text' and 'airline_sentiment' columns.")

    # Remove rows with missing sentiment labels
    df = df.dropna(subset=["airline_sentiment"])

    # Fill empty positive feedback with strong positive words
    positive_keywords = [
        "Amazing experience!", "Fantastic service!", "Loved my flight!", "super experience",

        "Best airline ever!", "Super smooth travel!", "Excellent service!","very good"

        "Best airline ever!", "Super smooth travel!", "Excellent service!","very good","nice"

    ]
    
    df.loc[(df["airline_sentiment"] == "positive") & (df["text"].isna()), "text"] = df["text"].fillna(lambda _: positive_keywords.pop(0) if positive_keywords else "Great flight!")

    # Apply text cleaning
    df["cleaned_text"] = df["text"].apply(clean_text)

    # Save cleaned data
    output_file = "c:/Users/TEJASRI REDDY.V/Desktop/ST17_Airline _sentiment/data/cleaned_airline_tweets.csv"
    df.to_csv(output_file, index=False)

    print(f"✅ Preprocessed data saved to {output_file}")

if __name__ == "__main__":
    clean_dataset()
    print(f" Preprocessed data saved to {output_file}")

if __name__ == "__main__":
    clean_dataset()

