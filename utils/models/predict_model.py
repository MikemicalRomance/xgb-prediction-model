from sklearn.metrics import accuracy_score, f1_score, recall_score

def model_performance(trained_model,test_data):
    predictions = trained_model.predict(test_data["X_test"])
    accuracy = accuracy_score(test_data["y_test"], predictions)
    f_score = f1_score(test_data["y_test"], predictions)
    recall = recall_score(test_data["y_test"], predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("F1 Score:", f_score)
    print("Recall:", recall)
    return accuracy, f1_score, recall_score


# print("predictions interation_range", xgb_cl.interation_range)
#predictions = xgb_cl.predict(X_test)
# TASK 1.5 log performance metrics F1 Score, Accuracy, Recall.
#accuracy = accuracy_score(y_test, predictions)
#f1_score = f1_score(y_test, predictions)
#recall = recall_score(y_test, predictions)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))
#print("F1 Score", f1_score)
#print("Recall", recall)