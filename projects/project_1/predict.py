import os
import pathlib
import pickle
import xgboost as xgb
import numpy as np

MODEL_FOLDER = pathlib.Path(__file__).parent.resolve() / 'models'
if "MODEL_FOLDER" in os.environ:
    MODEL_FOLDER = pathlib.Path(os.environ["MODEL_FOLDER"])

MODEL_NAME = os.getenv("MODEL_NAME", "model1.pickle")

def load_pickle_model():
    # Loads model from pickle file
    with open(MODEL_FOLDER / MODEL_NAME, 'rb') as f:
        model = pickle.load(f)

    with open(MODEL_FOLDER /'dv.pickle', 'rb') as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


def predict(model, vectorizer, input):
    # Predicts the input using the model
    input_transformed = vectorizer.transform(input)

    duser_val = xgb.DMatrix(input_transformed, feature_names=vectorizer.feature_names_)
    user_salary_log = model.predict(duser_val)[0]
    prediction = np.expm1(user_salary_log)

    return prediction


if __name__ == "__main__":
    input = {'age': 24.0,
             'gender': 'Male',
             'city': 'Munich',
             'position': 'Software Engineer',
             'experience_years': '5',
             'experience_years_germany': '2',
             'seniority_level': 'Middle',
             'vacation_days': '30',
             'employment_status': 'Full-time employee',
             'сontract_duration': 'Unlimited contract',
             'language_at_work': 'English',
             'company_size': '1000+',
             'company_type': 'Product'}
    model, vectorizer = load_pickle_model()
    prediction = predict(model, vectorizer, input)
    print(prediction)
