import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import re
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ✅ Step 1: Load & Preprocess Data
def clean_text(text):
    text = text.lower()
    text = re.sub(r"@\w+|#\w+|http\S+", "", text)  # Remove mentions, hashtags, URLs
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return " ".join(words)

# Load dataset
df = pd.read_csv("c:/Users/TEJASRI REDDY.V/Desktop/ST17_Airline _sentiment/data/cleaned_airline_tweets.csv")
df['cleaned_text'] = df['cleaned_text'].apply(clean_text)

# Convert sentiment labels to numeric values
sentiment_mapping = {"positive": 1, "neutral": 0, "negative": -1}
df["sentiment"] = df["airline_sentiment"].map(sentiment_mapping)

X = df["cleaned_text"]
y = df["sentiment"]

# ✅ Step 2: Convert Text into Word Embeddings
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")  # Handle unseen words with <OOV>
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
X_padded = pad_sequences(sequences, maxlen=50, padding="post")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# ✅ Step 3: Build LSTM Model
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=50),  # Embedding layer
    LSTM(64, return_sequences=True),  # LSTM Layer
    LSTM(32),
    Dense(32, activation="relu"),
    Dense(1, activation="tanh")  # Output layer (-1: Negative, 0: Neutral, 1: Positive)
])

model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])

# ✅ Step 4: Train Model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# ✅ Step 5: Function for Prediction
def predict_sentiment(text):
    text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=50, padding="post")
    prediction = model.predict(padded)
    if prediction > 0.3:
        return "Positive"
    elif prediction < -0.3:
        return "Negative"
    else:
        return "Neutral"

# Example Predictions
print(predict_sentiment("The flight was amazing!"))  # Expected: Positive
print(predict_sentiment("Terrible experience, never flying again!"))  # Expected: Negative
print(predict_sentiment("okay"))  # Expected: Neutral
