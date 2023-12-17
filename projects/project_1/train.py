import pickle
import re

import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import numpy as np


def fillna_mean(series: pd.Series):
    series_temp = pd.to_numeric(series, errors='coerce')
    mean = series_temp.mean()
    return series_temp.fillna(mean)


def extract_values_into_df(original_df: pd.DataFrame, column: str, prefix: str):
    original_column_low = original_df[column].fillna('').str.lower()
    col_arrays = original_column_low.str.split(',')

    values_unique = set()
    for arr in col_arrays:
        for item in arr:
            if item.strip() != '':
                values_unique.add(item.strip())

    new_columns = {}
    for value in values_unique:
        new_columns[f'{prefix}_{value}'] = original_column_low.str.contains(value, regex=False)

    return pd.DataFrame(new_columns)


def map_language(lang: str):
    # Since the survey mainly contains respondents from Germany (based on the city), I assumend that 50/50 or 'both' implies English & German.
    # So I've selected English as the main language
    if ('english' in lang) or lang == 'deuglisch' or lang == '50/50' or lang == 'both':
        return 'english'
    if lang == 'русский':
        return 'russian'

    return lang


def prepare(original_df: pd.DataFrame, include_main_tech=False):
    df = original_df.copy()
    df.columns = [re.sub('\W+','_',c.lower()) for c in df.columns]

    renamed_columns = {
        'total_years_of_experience': 'experience_years',
        'years_of_experience_in_germany': 'experience_years_germany',
        'your_main_technology_programming_language': 'main_technology',
        'other_technologies_programming_languages_you_use_often': 'other_technologies',
        'yearly_brutto_salary_without_bonus_and_stocks_in_eur': 'salary_brutto_without_bonuses',
        'yearly_bonus_stocks_in_eur': 'yearly_bonus',
        'number_of_vacation_days': 'vacation_days',
        'main_language_at_work': 'language_at_work',
        'position_': 'position'
    }
    df.rename(columns=renamed_columns, inplace=True)

    df['age'] = fillna_mean(df['age'])
    df['experience_years'] = fillna_mean(df['experience_years'])
    df['experience_years_germany'] = fillna_mean(df['experience_years_germany'])
    df['vacation_days'] = fillna_mean(df['vacation_days'])

    df['company_size'] = df['company_size'].str.lower().fillna('1000+')
    df['company_type'] = df['company_type'].str.lower().fillna('other')
    df['language_at_work'] = df['language_at_work'].fillna('english').str.lower().apply(map_language)
    df['сontract_duration'] = df['сontract_duration'].str.split(' ').str.get(0).str.lower().replace({'0': 'temporary'}).fillna('unlimited')

    df['full_employment'] = df['employment_status'].str.lower().str.contains('full').fillna(True)
    del df['employment_status']

    df['seniority_level'] = df['seniority_level'].str.lower().fillna('middle')
    df['position'] = df['position'].str.lower().fillna('software engineer')
    df['gender'] = df['gender'].str.lower().fillna('male')
    df['city'] = df['city'].str.lower()

    main_tech_df = extract_values_into_df(df, 'main_technology', 'mt')
    del df['main_technology']
    del df['other_technologies']

    salary_brutto = df['salary_brutto_without_bonuses'].astype(int) + pd.to_numeric(df['yearly_bonus'], errors='coerce').fillna(0)
    df['salary_brutto_log'] = np.log1p(salary_brutto)

    if include_main_tech:
        return pd.merge(df, main_tech_df, left_index=True, right_index=True)

    return df


def split(df):
    y = df['salary_brutto_log']
    del df['salary_brutto_log']

    return df, y


def train(train_df: pd.DataFrame, y_train: pd.Series):
    dv = DictVectorizer()

    train_dict = train_df.to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)
    xgb_r = xgb.XGBRegressor(objective='reg:linear', n_estimators=10)
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=dv.feature_names_)

    param = {"booster": "gblinear", "objective": "reg:squarederror"}
    xgb_r = xgb.train(params=param, dtrain=dtrain, num_boost_round=10)

    return dv, xgb_r


if __name__ == '__main__':
    df = pd.read_csv('IT Salary Survey EU  2020.csv')

    df = prepare(df, include_main_tech=False)

    train_df, y_train = split(df)
    dv, model = train(train_df, y_train)

    with open('models/model1.pickle', 'wb') as f:
        pickle.dump(model, f)

    with open('models/dv.pickle', 'wb') as f:
        pickle.dump(dv, f)

    print(f'Model trained and saved to models/model1.pickle')
    print(f'Vectorizer saved to models/dv.pickle')

