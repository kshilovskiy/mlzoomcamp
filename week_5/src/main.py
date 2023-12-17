import requests

from model_loader import load_pickle_model, predict


def predict_direct():
    input = {"job": "retired", "duration": 445, "poutcome": "success"}
    model, vectorizer = load_pickle_model()
    return predict(model, vectorizer, input)


def predict_via_request():
    url = "http://localhost:9696/predict"
    client = {"job": "unknown", "duration": 270, "poutcome": "failure"}

    return requests.post(url, json=client).json()


def predict_via_docker():
    url = "http://localhost:9696/predict"
    client = {"job": "retired", "duration": 445, "poutcome": "success"}

    return requests.post(url, json=client).json()


if __name__ == "__main__":
    prediction = predict_via_docker()
    print(prediction)
    # Result direct: 0.9019309332297606
    # HTTP Result {'prediction': 0.13968947052356817}
    # Docker Result {'prediction': 0.726936946355423}
