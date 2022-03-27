from sklearn.model_selection import train_test_split


def train_test_valid_split(
    dataframe, target_variable, train_size, test_size_remaining_data
):
    X = dataframe.drop(columns=[target_variable]).copy()
    y = dataframe[target_variable]
    # We want to split the data 60:20:20 for train:validation:test. Frist split the data in training and remaining dataset.
    X_train, X_remaining, y_train, y_remaining = train_test_split(
        X, y, train_size=train_size
    )
    # We want the validation and test sets size to each equal 20% each of overall data. Therefore set a test size that is 50% of remaining data.
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_remaining, y_remaining, test_size=test_size_remaining_data
    )
    return X_train, y_train, X_test, y_test, X_valid, y_valid


def train_model(
    model, training_data, validation_data, early_stopping_rounds, eval_metric
):
    trained_model = model.fit(
        training_data["X_train"],
        training_data["y_train"],
        early_stopping_rounds=early_stopping_rounds,
        eval_metric=eval_metric,
        eval_set=[(validation_data["X_valid"], validation_data["y_valid"])],
    )
    return trained_model


# xgb_cl = xgb.XGBClassifier()
# eval_set = [(X_valid, y_valid)]
# model = xgb_cl.fit(
#    X_train,
#    y_train,
#    early_stopping_rounds=20,
#    eval_metric="logloss",
#    eval_set=eval_set,
#    verbose=True,
# )
