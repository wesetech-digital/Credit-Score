import pytest
import pandas as pd
from scripts.transformers import DatetimeFeatureExtractor, LabelEncoderTransformer

# Test for DatetimeFeatureExtractor
def test_datetime_feature_extractor():
    # Sample data to test the DatetimeFeatureExtractor
    data = pd.DataFrame({
        'date_col': ['2021-01-01', '2021-02-02', '2021-03-03'],
        'other_col': [1, 2, 3]
    })

    # Instantiate the transformer with 'date_col' as the date column
    transformer = DatetimeFeatureExtractor(date_cols=['date_col'])
    
    # Fit and transform the data
    transformer.fit(data)
    transformed_data = transformer.transform(data)

    # Assert the new columns (month, day, year) exist and are correctly created
    assert 'date_col_month' in transformed_data.columns
    assert 'date_col_day' in transformed_data.columns
    assert 'date_col_year' in transformed_data.columns

    # Assert the original date column has been dropped
    assert 'date_col' not in transformed_data.columns

    # Check if the extracted month, day, and year are correct
    assert transformed_data['date_col_month'].iloc[0] == 1
    assert transformed_data['date_col_day'].iloc[0] == 1
    assert transformed_data['date_col_year'].iloc[0] == 2021

    # Check that the other columns remain unchanged
    assert transformed_data['other_col'].iloc[0] == 1

# Test for LabelEncoderTransformer
def test_label_encoder_transformer():
    # Sample data to test the LabelEncoderTransformer
    data = pd.DataFrame({
        'category_col': ['A', 'B', 'A', 'C', 'B'],
        'other_col': [1, 2, 3, 4, 5]
    })

    # Instantiate the transformer with 'category_col' as the categorical column
    transformer = LabelEncoderTransformer(cat_cols=['category_col'])
    
    # Fit and transform the data
    transformer.fit(data)
    transformed_data = transformer.transform(data)

    # Assert the original column has been encoded correctly
    assert transformed_data['category_col'].iloc[0] == 0  # 'A' should be encoded as 0
    assert transformed_data['category_col'].iloc[1] == 1  # 'B' should be encoded as 1
    assert transformed_data['category_col'].iloc[2] == 0  # 'A' should be encoded as 0
    assert transformed_data['category_col'].iloc[3] == 2  # 'C' should be encoded as 2
    assert transformed_data['category_col'].iloc[4] == 1  # 'B' should be encoded as 1

    # Assert the other column remains unchanged
    assert transformed_data['other_col'].iloc[0] == 1

def test_label_encoder_transformer_unknown_category():
    # Sample data to test handling of unknown categories
    data_train = pd.DataFrame({
        'category_col': ['A', 'B', 'A', 'C'],
        'other_col': [1, 2, 3, 4]
    })

    data_test = pd.DataFrame({
        'category_col': ['D', 'E'],
        'other_col': [5, 6]
    })

    transformer = LabelEncoderTransformer(cat_cols=['category_col'])

    # Fit the transformer on training data
    transformer.fit(data_train)

    # Transform both training and test data
    transformed_train = transformer.transform(data_train)
    transformed_test = transformer.transform(data_test)

    # Assert transformations
    assert 'category_col' in transformed_test.columns
    assert transformed_test['category_col'].iloc[0] == -1  # Unseen category 'D' should be encoded as -1
    assert transformed_test['category_col'].iloc[1] == -1  # Unseen category 'E' should be encoded as -1
