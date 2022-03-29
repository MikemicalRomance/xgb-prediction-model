# Script to download dataset
import pandas as pd

pd.set_option("display.max_rows", None)


def get_dataset(url: str):
    df = pd.read_csv(url)
    return df
