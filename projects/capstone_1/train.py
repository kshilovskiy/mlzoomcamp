import pickle
import re

import pandas as pd
import xgboost as xgb

from common import TARGET, FEATURES, prepare

SEED = 42


def split(df):
    X = df[FEATURES]
    y = df[TARGET]

    return X, y


def train(X: pd.DataFrame, y: pd.Series):
    xgb_model = xgb.XGBClassifier(
        learning_rate=0.2,
        max_depth=9,
        n_estimators=200,
        seed=SEED
    )

    xgb_model.fit(X, y)
    return xgb_model


if __name__ == '__main__':
    df = pd.read_csv('train_big.csv')

    df = prepare(df)

    df_train, y_train = split(df)
    model = train(df_train, y_train)

    model_path = 'models/model1.pickle'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f'Model trained and saved to {model_path}')
