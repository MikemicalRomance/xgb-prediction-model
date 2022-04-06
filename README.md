# xgb-prediction-model 
XGBoost prediction model in python. 
```XGB-PREDICTION-MODEL 
├── artifacts   
│         └── model    <- saved model 
├── output             <- output for resulting csv
│   └── results.csv  
│                     
├── utils              <- utils for processing data, building, training, and evaluating models 
│   ├── __init__.py    <- Makes src a Python module 
│   │
│   ├── data           <- Scripts to fetch data 
│   │   └── get_data.py 
│   │
│   ├── features       <- Scripts to encode features  
│   │   └── encode_data.py 
│   │
│   ├── models         <- Scripts to train models and then use trained models to make predictions 
│   │   │                   
│   │   ├── predict_model.py  
│   │   └── train_model.py  
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py  
| 
├── variables   
|   ├── __init__.py    <- Makes variables a Python module   
│   └── variables.py   <- Script to store shared project variables, binary and label column names.  
| 
|── pipfile            <-  package requirements for application.  
|
|── pipfile.lock       <- which specific version of packages should be used. Run: `pipenv install`. 
| 
├── README.md          <- The top-level README for developers using this project. 
│ 
├── task_1_model_creation.py          <- script to train and save model.  
| 
├── task_2_model_output.py         <- script to load, make predictions and save results.  
| 
└── test_prediction.py            <- test script to check all required columns are present in results_df. 
```
# XGBoost prediction model

Python program to predict adoption of pets based of historic data.

a python script to:
1. Read the input from `gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv` and load it in a Pandas Dataframe.
2. Split the dataset in 3 splits: train, validation, test with ratio of 60 (train) / 20 (validation) / 20 (test)
3. Perform any feature engineering  you might find useful to enable training. It's not required that you create new features. 
4. Train an ML model using XGB to predict whether a pet will be adopted or not `Adopted` is the target feature. You will need to use the validation to assess early stopping. You won't need to hypertune any parameter, the default parameters will be sufficient, with the exception of the number of trees which gets tuned by the early stopping mechanism.
5. The script needs to log to the user the performances of the model in the test set in terms of F1 Score, Accuracy, Recall.

Save the model into `artifacts/model` and make sure the folder is <b>not</b> git ignored.


## Part 2
Write a python script to:
1. Load the data from `gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv`
2. Uses the model you trained in the previous step to score all the rows in the CSV, excluding of course the header.
3. Save the output into `output/results.csv` and make sure all files in the `output/` directory is git ignored.
4. Add a unit test to the prediction function.


The output needs to follow the following format:
```
Type,Age,Breed1,Gender,Color1,Color2,MaturitySize,FurLength,Vaccinated,Sterilized,Health,Fee,PhotoAmt,Adopted,Adopted_prediction
Cat,3,Tabby,Male,Black,White,Small,Short,No,No,Healthy,100,1,Yes, No
```


