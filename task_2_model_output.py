from utils import (
    get_dataset,
    binary_encode,
    label_encode,
    binary_decode,
    predict_function
)
from variables import binary_columns, binary_positive_values, label_columns
import xgboost as xgb
# TASK 2.1 Fetch data
petfinder_df = get_dataset(
    "gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv"
)

loaded_model = xgb.XGBClassifier()
loaded_model.load_model("./artifacts/model/model.json")
binary_encoded_df = binary_encode(petfinder_df, binary_columns, binary_positive_values)
fully_encoded_df = label_encode(binary_encoded_df, label_columns)


encoded_predictions = predict_function(trained_model=loaded_model,unencoded_data=petfinder_df,encoded_data=fully_encoded_df,target_variable="Adopted",predicted_variable="Adopted_prediction")   
output_dataframe = binary_decode(encoded_predictions,['Adopted_prediction'], ['Yes'],['No'])

output_dataframe.to_csv('./output/results.csv', index= False)

