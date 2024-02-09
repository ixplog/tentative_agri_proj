import numpy as np
import pandas as pd
import yaml
import logging

logger = logging.getLogger(__name__)


### Paramaters read

with open('conf/base/parameters.yml') as params:
    config = yaml.safe_load(params)
    label_cols = config["label_cols"]

    train_data_split = config["train_data_split"]
    val_data_split = config["val_data_split"]


### Pipeline nodes

def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the data.

    Args:
        df: Raw data.
    Returns:
        Preprocessed data
    """
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values("DateTime")
    df = df.set_index('DateTime')
    df = df.drop(["Record"], axis=1)
    return df

def split_data(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """Splits data into a training, a validation and a test dataset

    Args:
        df: Cleaned data.
    Returns:
        train_df: training dataset.
        val_df: validation dataset.
        test_df: test dataset.
    """
    n = len(df)
    train_df = df[0:int(n*train_data_split)]
    val_df = df[int(n*train_data_split):int(n*val_data_split)]
    test_df = df[int(n*val_data_split):]

    return train_df, val_df, test_df

def compute_mean_std(df):
    """Computes mean and std of an input DataFrame

    Args:
        df: input dataset.
    Returns:
        mean: mean of input dataset.
        std: std of input dataset.
    """
    mean = df.mean()
    std = df.std()
    return mean, std

def normalise_data(df, mean, std) -> pd.DataFrame:
    """Normalises data according to training dataset

    Args:
        df: input dataset.
        mean: mean parameter.
        std: std parameter.
    Returns:
        df: dataset normalised.
    """
    df = (df - mean) / std
    return df