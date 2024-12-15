# custom_transformers.py
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.preprocessing import  LabelEncoder


class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, date_cols):
        self.date_cols = date_cols

    def fit(self, X, y=None):
        # No fitting needed for this transformer
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.date_cols:
            X[col] = pd.to_datetime(X[col], errors='coerce')  # Convert to datetime, coerce invalid dates to NaT
            # Extract year, month, and day
            X[col + '_year'] = X[col].dt.year.fillna(0)  # Replace NaT years with 0
            X[col + '_month'] = X[col].dt.month.fillna(0)  # Replace NaT months with 0
            X[col + '_day'] = X[col].dt.day.fillna(0)  # Replace NaT days with 0
        return X.drop(columns=self.date_cols, errors='ignore')  # Drop original date columns



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
        unseen_labels = {}

        for col, le in self.encoders.items():
            # Find unseen labels
            unseen = [x for x in X[col].unique() if x not in le.classes_]
            if unseen:
                unseen_labels[col] = unseen
                print(f"Warning: Unseen labels found in column '{col}': {unseen}")

            # Apply label encoding for known labels, and default to -1 for unseen labels
            X[col] = X[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        if unseen_labels:
            print(f"Unseen labels handled with default value -1: {unseen_labels}")

        return X