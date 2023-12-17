import re

import pandas as pd

FEATURES = ['age',
            'height_cm_',
            'weight_kg_',
            'waist_cm_',
            'eyesight_left_',
            'eyesight_right_',
            'hearing_left_',
            'hearing_right_',
            'systolic',
            'relaxation',
            'fasting_blood_sugar',
            'cholesterol',
            'triglyceride',
            'hdl',
            'ldl',
            'hemoglobin',
            'urine_protein',
            'serum_creatinine',
            'ast',
            'alt',
            'gtp',
            'dental_caries']

TARGET = 'smoking'


def prepare(df_original: pd.DataFrame):
    df = df_original.copy()
    df.columns = [re.sub('\W+', '_', c.lower()) for c in df.columns]
    return df
