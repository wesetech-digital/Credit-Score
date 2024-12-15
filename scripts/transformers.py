# custom_transformers.py
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.preprocessing import  LabelEncoder


class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, date_cols):
        self.date_cols = date_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.date_cols:
            X[col] = pd.to_datetime(X[col], errors='coerce')
            X[col + '_month'] = X[col].dt.month
            X[col + '_day'] = X[col].dt.day
            X[col + '_year'] = X[col].dt.year
        return X.drop(columns=self.date_cols, errors='ignore')

class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols):
        self.cat_cols = cat_cols
        self.encoders = {}

    def fit(self, X, y=None):
        for col in self.cat_cols:
            le = LabelEncoder()
            le.fit(X[col])
            self.encoders[col] = le
        return self

    def transform(self, X):
        X = X.copy()
        for col, le in self.encoders.items():
            X[col] = le.transform(X[col])
        return X
