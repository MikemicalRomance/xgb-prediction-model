from utils import (
    get_dataset,
    binary_encode,
    label_encode,
    train_test_valid_split,
    train_xgb_classifier,
    model_performance,
    eval_metric_plot
)
from variables import binary_columns, binary_positive_values, label_columns

# TASK 1.1 Fetch data set
petfinder_df = get_dataset(
    "gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv"
)
## initial EDA; data looks pretty clean all 14 columns have 11,537 non null values, distribution of cat and dog is acceptable
## TODO: clean data breed Tortoiseshell is not a breed

# TASK 1.3 Feature engineering:
# Numerical columns: Age, Fee, PhotoAmt.
# Columns for Binary Encoding: Type, Gender, Adopted.
# Columns for Label Encoding: Breed1, Color1, Color2, MaturitySize, Furlength, Vaccinated, Sterilized, Health.

binary_encoded_df = binary_encode(petfinder_df, binary_columns, binary_positive_values)

fully_encoded_df = label_encode(binary_encoded_df, label_columns)

X_train, y_train, X_test, y_test, X_valid, y_valid = train_test_valid_split(
    dataframe=fully_encoded_df,
    target_variable="Adopted",
    train_size=0.6,
    test_size_remaining_data=0.5,
)

training_data = {
    "X_train": X_train,
    "y_train": y_train,
}

test_data = {
    "X_test": X_test,
    "y_test": y_test,
}

validation_data = {
    "X_valid": X_valid,
    "y_valid": y_valid,
}

# TASK 1.4 Train XGB Model:
trained_xgb_classifier = train_xgb_classifier(
    training_data=training_data,
    validation_data=validation_data,
    early_stopping_rounds=20,
    eval_metric="logloss",
)

# TASK 1.5 log performance metrics F1 Score, Accuracy, Recall.
model_performance(trained_model = trained_xgb_classifier,test_data=test_data)
print("best iteration", trained_xgb_classifier.best_iteration)
# Save model to artifacts dir
trained_xgb_classifier.save_model("artifacts/model/model.json")
#optional plots
eval_metric_plot(trained_xgb_classifier.evals_result(),'logloss')
