import os
import pathlib
import pickle

import pandas as pd
import xgboost

from common import prepare, FEATURES

MODEL_FOLDER = pathlib.Path(__file__).parent.resolve() / 'models'
if "MODEL_FOLDER"in os.environ:
    MODEL_FOLDER = pathlib.Path(os.environ["MODEL_FOLDER"])

MODEL_NAME = os.getenv("MODEL_NAME", "model1.pickle")


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


if __name__ == "__main__":
    p_input = {"id": 1113.0,
               "age": 40.0,
               "height(cm)": 175.0,
               "weight(kg)": 75.0,
               "waist(cm)": 85.0,
               "eyesight(left)": 1.5,
               "eyesight(right)": 1.5,
               "hearing(left)": 1.0,
               "hearing(right)": 1.0,
               "systolic": 116.0,
               "relaxation": 74.0,
               "fasting blood sugar": 92.0,
               "Cholesterol": 179.0,
               "triglyceride": 206.0,
               "HDL": 41.0,
               "LDL": 97.0,
               "hemoglobin": 17.1,
               "Urine protein": 1.0,
               "serum creatinine": 1.1,
               "AST": 23.0,
               "ALT": 35.0,
               "Gtp": 56.0,
               "dental caries": 0.0,
               "smoking": 1.0}
    # p_input = {"id": 1111.0,
    #          "age": 50.0,
    #          "height(cm)": 160.0,
    #          "weight(kg)": 60.0,
    #          "waist(cm)": 76.6,
    #          "eyesight(left)": 1.2,
    #          "eyesight(right)": 1.2,
    #          "hearing(left)": 1.0,
    #          "hearing(right)": 1.0,
    #          "systolic": 120.0,
    #          "relaxation": 80.0,
    #          "fasting blood sugar": 86.0,
    #          "Cholesterol": 155.0,
    #          "triglyceride": 72.0,
    #          "HDL": 47.0,
    #          "LDL": 94.0,
    #          "hemoglobin": 14.9,
    #          "Urine protein": 1.0,
    #          "serum creatinine": 1.0,
    #          "AST": 23.0,
    #          "ALT": 14.0,
    #          "Gtp": 23.0,
    #          "dental caries": 0.0,
    #          "smoking": 0.0}
    loaded_model = _load_pickle_model()
    prediction = _predict(loaded_model, p_input)
    print(prediction)
