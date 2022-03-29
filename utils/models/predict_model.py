from sklearn.metrics import accuracy_score, f1_score, recall_score

# assess model performance in terms of accuracy,1 Score and recall
def model_performance(trained_model, test_data):
    predictions = trained_model.predict(test_data["X_test"])
    accuracy = accuracy_score(test_data["y_test"], predictions)
    f_score = f1_score(test_data["y_test"], predictions)
    recall = recall_score(test_data["y_test"], predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("F1 Score:", f_score)
    print("Recall:", recall)
    return accuracy, f1_score, recall_score

# make predictions given trained model and data
def predict_function(
    trained_model,
    unencoded_data,
    encoded_data,
    target_variable: str,
    predicted_variable: str,
):
    prediction_data = encoded_data.drop(columns=[target_variable]).copy()
    predictions = trained_model.predict(prediction_data)
    output_dataframe = unencoded_data.copy()
    output_dataframe[predicted_variable] = predictions
    return output_dataframe
