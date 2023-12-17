import os
import pathlib
import pickle

MODEL_FOLDER = pathlib.Path(__file__).parent.resolve() / 'resources'
if os.getenv("MODEL_FOLDER") is not None:
    MODEL_FOLDER = pathlib.Path(os.getenv("MODEL_FOLDER"))

MODEL_NAME = os.getenv("MODEL_NAME", "model1.bin")

def load_pickle_model():
    # Loads model from pickle file
    with open(MODEL_FOLDER / MODEL_NAME, 'rb') as f:
        model = pickle.load(f)

    with open(MODEL_FOLDER /'dv.bin', 'rb') as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


def predict(model, vectorizer, input):
    # Predicts the input using the model
    input_transformed = vectorizer.transform(input)
    prediction = model.predict_proba(input_transformed)[0, 1]
    return prediction


if __name__ == "__main__":
    input = {"job": "retired", "duration": 445, "poutcome": "success"}
    model, vectorizer = load_pickle_model()
    prediction = predict(model, vectorizer, input)
    print(prediction)
