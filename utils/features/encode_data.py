# Script to encode Binary and Categorical Data
from sklearn.preprocessing import LabelEncoder
def binary_encode(df, columns, positive_values):
    df=df.copy()
    for column, positive_value in zip(columns, positive_values):
        df[column] = df[column].apply(lambda x: 1 if x == positive_value else 0)
    return df


def binary_decode(df, columns, positive_values, negative_values):
    df = df.copy()
    for column, positive_value,negative_value in zip(columns, positive_values, negative_values):
        df[column] = df[column].apply(lambda x: positive_value if x == 1 else negative_value)
    return df


def label_encode(df, columns):
    df =df.copy()
    for column in columns:
        df[column] = LabelEncoder().fit_transform(df[column])
    return df
