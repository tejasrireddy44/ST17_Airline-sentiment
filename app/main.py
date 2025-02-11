from flask import Flask, request, jsonify
from scripts.predict import predict_sentiment

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    sentiment = predict_sentiment(text)
    return jsonify({"sentiment": sentiment})

if __name__ == "__main__":
    app.run(debug=True)
