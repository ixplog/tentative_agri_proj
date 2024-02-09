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


#-----------------------------------------------------------------------


# class MyTFModel(Model):
#     def __init__(self):
#         super().__init__()
#         self.lstm = LSTM(units=100)
#         self.dense1 = Dense(128, activation='relu')
#         self.dense2 = Dense(10)
#         self.dense3 = Dense(2)
    
#     def call(self, x):
#         x = self.lstm(x)
#         x = self.dense1(x)
#         x = self.dense2(x)
#         return self.dense3(x)


def define_model() -> str:
    """Defines a model

    Returns:
        A model description
    """

    model = tf.keras.Sequential()
    model.add(LSTM(units=100))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10))
    model.add(Dense(2))

    loss = tf.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()
    metrics = [tf.metrics.MeanAbsoluteError()]

    #model = MyTFModel()
    model.compile(optimizer, loss, metrics=metrics)

    # TO BE READ FROM CONFIG
    input_shape = (None, 10, 5)
    model.build(input_shape)

    # TO BE READ FROM CONFIG
    path = "data/06_models/tensorflow_lstm"
    model.save(path, overwrite=True)

    return model.to_json() # Kedro only saves serialisable objects


#-----------------------------------------------------------------------


def get_windowed_data(df: pd.DataFrame) -> (np.array, np.array):
    """Prepares the data for input to an LSTM model

    Args:
        df: cleaned and normalised data.
    Returns:
        X, Y pair of windowed data
    """
    # TO BE READ FROM CONFIG
    window_size = 10
    label_cols = ["RH.percent", "Tair.C"]

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


def train_model(X_train: np.array, Y_train: np.array, X_val: np.array, Y_val: np.array, model_description) -> History:
    """Trains the model

    Args:
        X_train: independent test data.
        Y_train: dependent test data.
        X_val: independent validation data.
        Y_val: dependent validation data.
        model_description: the description of the model to be trained.
    Returns:
        History object.
    """
    # read model_description and read model accordingly
    # TODO

    # TO BE READ FROM CONFIG
    path = "data/06_models/tensorflow_lstm"
    model = tf.keras.saving.load_model(path)

    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=15)

    # TO BE READ FROM CONFIG
    path = "data/06_models/tensorflow_lstm_trained"
    model.save(path, overwrite=True)

    return history


def evaluate_model(X_test: np.array, Y_test: np.array, history) -> float:
    """Evaluates the model

    Args:
        X_test: independent test data.
        Y_test: dependent test data.
        model: model to be evaluated.
    Returns:
        Scalar test loss.
    """
    # TO BE READ FROM CONFIG
    path = "data/06_models/tensorflow_lstm_trained"
    model = tf.keras.saving.load_model(path)

    test_result = model.evaluate(X_test, Y_test)

    # TODO: compare with history

    return test_result