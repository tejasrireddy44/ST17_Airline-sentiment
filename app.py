import nltk
nltk.download('punkt_tab')
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Initialize Flask app
app = Flask(__name__)

# Download NLTK resources if not available
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

# Preload stopwords & lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Cleans and preprocesses text for sentiment analysis."""
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
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)

def analyze_sentiment(text):
    """Dummy function for sentiment analysis (replace with ML model later)."""
    text = clean_text(text)
    positive_words = ["good", "amazing", "fantastic", "great", "excellent", "loved"]
    negative_words = ["bad", "terrible", "worst", "hate", "awful", "sad"]

    if any(word in text for word in positive_words):
        return "positive"
    elif any(word in text for word in negative_words):
        return "negative"
    else:
        return "neutral"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    user_text = request.form.get("feedback")

    if not user_text.strip():
        return redirect(url_for("home"))

    sentiment = analyze_sentiment(user_text)
    
    return redirect(url_for("result", sentiment=sentiment))

@app.route("/result")
def result():
    sentiment = request.args.get("sentiment", "neutral")
    return render_template("result.html", sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)