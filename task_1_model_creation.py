import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as pyplot

pd.set_option("display.max_rows", None)

# TASK 1.1 Fetch data & data cleaning
petfinder_df = pd.read_csv(
    "gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv"
)
print(petfinder_df.head(5))
## initial EDA; data looks pretty clean all 14 columns have 11,537 non null values, distribution of cat and dog is acceptable
## TODO: data clean breed Tortoiseshell is not a breed
# Breed1 value count
# 0 4
# Tortoiseshell  35

# TASK 1.3 Feature engineering:
# Numerical: Age, Fee, PhotoAmt
# Binary Encode: Type: [Cat,Dog], Gender:[Male, Female], Adopted: [Yes, No]
def binary_encode(df, columns, positive_values):
    df.copy()
    for column, positive_value in zip(columns, positive_values):
        df[column] = df[column].apply(lambda x: 1 if x == positive_value else 0)
    return df


binary_columns = ["Type", "Gender", "Adopted"]
binary_positive_values = ["Cat", "Female", "Yes"]
binary_encoded_df = binary_encode(petfinder_df, binary_columns, binary_positive_values)

# Label Encode: Breed1, Color1, Color2, MaturitySize, Furlength, Vaccinated, Sterilized, Health
def label_encode(df, columns):
    df.copy()
    for column in columns:
        df[column] = LabelEncoder().fit_transform(df[column])
    return df


label_columns = [
    "Breed1",
    "Color1",
    "Color2",
    "MaturitySize",
    "FurLength",
    "Vaccinated",
    "Sterilized",
    "Health",
]
label_encoded_df = label_encode(binary_encoded_df, label_columns)

# TASK 1.2 Train Test Split:
X = label_encoded_df.drop(columns=["Adopted"]).copy()
y = label_encoded_df["Adopted"]
# We want to split the data 60:20:20 for train:validation:test. Frist split the data in training and remaining dataset.
train_size = 0.6
X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.6)

# We want the validation and test sets size to each equal 20% each of overall data. Therefore set a test size that is 50% of remaining data.
test_size = 0.5
X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)

# TASK 1.4 Train XGB Model:
# Train
xgb_cl = xgb.XGBClassifier()
eval_set = [(X_valid, y_valid)]
model = xgb_cl.fit(
    X_train,
    y_train,
    early_stopping_rounds=20,
    eval_metric="logloss",
    eval_set=eval_set,
    verbose=True,
)
# Predict
predictions = xgb_cl.predict(X_test, ntree_limit=model.best_ntree_limit)

# TASK 1.5 log performance metrics F1 Score, Accuracy, Recall.
accuracy = accuracy_score(y_test, predictions)
f1_score = f1_score(y_test, predictions)
recall = recall_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("F1 Score", f1_score)
print("Recall", recall)
# Save model to artifacts dir
model.save_model("artifacts/model/model.json")

## optional plots
results = model.evals_result()
epochs = len(results["validation_0"]["logloss"])
x_axis = range(0, epochs)
# plot log loss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results["validation_0"]["logloss"], label="validation set")
# ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
pyplot.ylabel("Log Loss")
pyplot.title("XGBoost Log Loss")
pyplot.show()
# plot classification error
fig, ax = pyplot.subplots()
ax.plot(x_axis, results["validation_0"]["logloss"], label="validation set")
# ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
pyplot.ylabel("Classification Error")
pyplot.title("XGBoost Classification Error")
pyplot.show()
