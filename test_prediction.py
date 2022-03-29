import pytest
import pandas as pd
import xgboost as xgb
from .utils import (
    get_dataset,
    binary_encode,
    label_encode,
    binary_decode,
    predict_function,
)
from .variables import binary_columns, binary_positive_values, label_columns

# TASK 2.4 add unit test to prediction function
petfinder_df = get_dataset(
    "gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv"
)

loaded_model = xgb.XGBClassifier()
loaded_model.load_model("./artifacts/model/model.json")
binary_encoded_df = binary_encode(petfinder_df, binary_columns, binary_positive_values)
fully_encoded_df = label_encode(binary_encoded_df, label_columns)


def test_predict_function():
    # call function
    actual_columns = list(
        predict_function(
            trained_model=loaded_model,
            unencoded_data=petfinder_df,
            encoded_data=fully_encoded_df,
            target_variable="Adopted",
            predicted_variable="Adopted_prediction",
        ).columns.values
    )
    # expectation
    expected_columns = df_columns = [
        "Type",
        "Age",
        "Breed1",
        "Gender",
        "Color1",
        "Color2",
        "MaturitySize",
        "FurLength",
        "Vaccinated",
        "Sterilized",
        "Health",
        "Fee",
        "PhotoAmt",
        "Adopted",
        "Adopted_prediction",
    ]
    # assertion
    assert actual_columns == expected_columns
