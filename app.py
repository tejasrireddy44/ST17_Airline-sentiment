import os
import pickle
import re
import string
from flask import Flask, render_template, request, redirect, url_for, jsonify
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize Flask app
app = Flask(__name__)

# Download NLTK resources (only needed once)
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

# Load stopwords & lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ✅ Set relative model path (for Render deployment)
model_path = os.path.join(os.path.dirname(__file__), "models")

# ✅ Check if model files exist before loading
sentiment_model_file = os.path.join(model_path, "sentiment_model.pkl")
tfidf_vectorizer_file = os.path.join(model_path, "tfidf_vectorizer.pkl")

if not os.path.exists(sentiment_model_file):
    raise FileNotFoundError(f"Model file not found: {sentiment_model_file}")

if not os.path.exists(tfidf_vectorizer_file):
    raise FileNotFoundError(f"Vectorizer file not found: {tfidf_vectorizer_file}")

# ✅ Load model & vectorizer
with open(sentiment_model_file, "rb") as f:
    model = pickle.load(f)

with open(tfidf_vectorizer_file, "rb") as f:
    vectorizer = pickle.load(f)

def clean_text(text):
    """Cleans and preprocesses text for sentiment analysis."""
    if not isinstance(text, str):
        return ""

    text = re.sub(r"@\w+|#\w+|http\S+", "", text.lower())  # Remove mentions, hashtags, URLs
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    tokens = [lemmatizer.lemmatize(word) for word in word_tokenize(text) if word not in stop_words]

    return " ".join(tokens)

def analyze_sentiment_ml(text):
    """Predict sentiment using the trained ML model with a confidence threshold."""
    cleaned_text = clean_text(text)
    transformed_text = vectorizer.transform([cleaned_text])
    
    probs = model.predict_proba(transformed_text)[0]
    predicted_label = model.classes_[probs.argmax()]

    # ✅ Set a threshold to avoid misclassification of weak predictions
    if probs.max() < 0.6:
        return "neutral"
    
    return predicted_label

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handles both JSON and form-based predictions."""
    if request.content_type == "application/json":
        data = request.get_json()
        text = data.get("text", "")
    else:
        text = request.form.get("feedback", "")

    if not text.strip():
        return redirect(url_for("home"))

    sentiment = analyze_sentiment_ml(text)  

    if request.content_type == "application/json":
        return jsonify({"sentiment": sentiment})

    return redirect(url_for("result", sentiment=sentiment))

@app.route("/result")
def result():
    sentiment = request.args.get("sentiment", "neutral")
    return render_template("result.html", sentiment=sentiment)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
