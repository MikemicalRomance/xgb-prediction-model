# Script to download dataset
import pandas as pd


def get_dataset(url: str):
    df = pd.read_csv(url)
    return df
