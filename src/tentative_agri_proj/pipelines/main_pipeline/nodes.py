import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import History


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
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n*0.9):]
    return train_df, val_df, test_df

def normalise_data(train_df, val_df, test_df):
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std
    return train_df, val_df, test_df


#-----------------------------------------------------------------------


class MyTFModel(Model):
    def __init__(self):
        super().__init__()
        self.lstm = LSTM(units=100)
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(10)
        self.dense3 = Dense(2)
    
    def call(self, x):
        x = self.lstm(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)


def define_model() -> MyTFModel:
    """Defines a model

    Returns:
        A model
    """
    loss = tf.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()
    metrics = [tf.metrics.MeanAbsoluteError()]

    model = MyTFModel()
    model.compile(optimizer, loss, metrics=metrics)

    return model


#-----------------------------------------------------------------------


def get_windowed_data(df: pd.DataFrame, window_size, label_cols) -> (np.array, np.array):
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


def train_model(X_train: np.array, Y_train: np.array, X_val: np.array, Y_val: np.array, model: MyTFModel) -> History:
    """Trains the model

    Args:
        X_train: independent test data.
        Y_train: dependent test data.
        X_val: independent validation data.
        Y_val: dependent validation data.
        model: the model to be trained.
    Returns:
        History object.
    """
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=15)
    return model, history


def evaluate_model(X_test: np.array, Y_test: np.array, model: MyTFModel) -> float:
    """Evaluates the model

    Args:
        X_test: independent test data.
        Y_test: dependent test data.
        model: model to be evaluated.
    Returns:
        Scalar test loss.
    """
    return model.evaluate(X_test, Y_test)


def save_model(model: MyTFModel, name: str):
    """Saves the model

    Args:
        model: th TF model to be saved.
        name: the name of the saved file.
    """
    model.export(f"../data/06_models/{name}")
    return