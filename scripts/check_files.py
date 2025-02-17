import os

model_path = "c:/Users/TEJASRI REDDY.V/Desktop/ST17_Airline _sentiment/models/sentiment_model.pkl"
vectorizer_path = "c:/Users/TEJASRI REDDY.V/Desktop/ST17_Airline _sentiment/models/tfidf_vectorizer.pkl"

print(f"Model Exists: {os.path.exists(model_path)}")
print(f"Vectorizer Exists: {os.path.exists(vectorizer_path)}")
