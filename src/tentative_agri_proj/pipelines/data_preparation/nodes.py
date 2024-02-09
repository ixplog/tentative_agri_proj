import numpy as np
import pandas as pd
import yaml

# Paramaters read

with open('conf/base/parameters.yml') as params:
    config = yaml.safe_load(params)

    train_data_split = config["train_data_split"]
    val_data_split = config["val_data_split"]
    window_size = config["window_size"]
    label_cols = config["label_cols"]


# Pipeline nodes

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

def normalise_data(train_df, val_df, test_df) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """Normalises data according to training dataset

    Args:
        train_df: training dataset.
        val_df: validation dataset.
        test_df: test dataset.
    Returns:
        train_df_norm: training dataset normalised.
        val_df_norm: validation dataset normalised.
        test_df_norm: test dataset normalised.
    """
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    return train_df, val_df, test_df


def get_windowed_data(df: pd.DataFrame) -> (np.array, np.array):
    """Prepares the data for input to an LSTM model

    Args:
        df: cleaned and normalised data.
    Returns:
        X, Y pair of windowed data
    """
    label_to_index = {name: i for i, name in enumerate(df.columns)}
    label_cols_indices = []
    if isinstance(label_cols, list):
        for col in label_cols:
            label_cols_indices.append(label_to_index[col])
    else:
        label_cols_indices.append(label_to_index[label_cols])
        
    X = []
    Y = []
    for i in range(len(df) - window_size):
        X.append(df.iloc[i:i+window_size].values)
        Y.append(df.iloc[i+window_size, label_cols_indices].values)

    return np.array(X), np.array(Y)