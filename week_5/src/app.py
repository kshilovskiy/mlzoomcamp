# Flask app serving model predictions via API
from flask import Flask, request, jsonify
from model_loader import load_pickle_model, predict

app = Flask(__name__)

model, vectorizer = load_pickle_model()


@app.route("/ping")
def hello():
    return "pong"

# Post requests to predict with a json body
@app.route("/predict", methods=["POST"])
def predict_api():
    #  Get input from request body
    input = request.json
    prediction = predict(model, vectorizer, input)
    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(host='localhost', port=8000, debug=True)
