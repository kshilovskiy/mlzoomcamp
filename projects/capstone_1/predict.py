import os
import pathlib
import pickle

import pandas as pd
import xgboost
from flask import Flask, request, jsonify

from common import prepare, FEATURES

MODEL_FOLDER = pathlib.Path(__file__).parent.resolve() / 'models'
if "MODEL_FOLDER" in os.environ:
    MODEL_FOLDER = pathlib.Path(os.environ["MODEL_FOLDER"])

MODEL_NAME = os.getenv("MODEL_NAME", "model1.pickle")

app = Flask(__name__)


def _load_pickle_model():
    # Loads model from pickle file
    with open(MODEL_FOLDER / MODEL_NAME, 'rb') as f:
        return pickle.load(f)


def _predict(model: xgboost.XGBClassifier, pred_input):
    # Predicts the input using the model
    input_transformed = _prepare_input(pred_input)

    predictions = model.predict_proba(input_transformed)

    return predictions[:, 1]


def _prepare_input(pred_input: dict):
    df = pd.DataFrame([pred_input])
    df = prepare(df)
    df = df[FEATURES]
    return df


model = _load_pickle_model()


@app.route("/ping")
def hello():
    return "pong"


# Post requests to predict with a json body
@app.route("/predict", methods=["POST"])
def predict_api():
    #  Get input from request body
    prediction = _predict(model, request.json)
    return jsonify({"prediction": str(prediction)})


if __name__ == "__main__":
    app.run(host='localhost', port=8000, debug=True)