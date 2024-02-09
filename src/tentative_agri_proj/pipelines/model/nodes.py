import numpy as np
import pandas as pd
import yaml

import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import History

import logging
logger = logging.getLogger(__name__)


### Paramaters read

model_name = "TF_LSTM_3xDense"

with open('conf/base/parameters.yml') as params:
    config = yaml.safe_load(params)
    label_cols = config["label_cols"]

    input_shape = (None, config["models"][model_name]["window_size"], config["num_features"])

    untrained_tf_lstm_model = config["models"][model_name]["untrained_tf_lstm_model"]
    trained_tf_lstm_model = config["models"][model_name]["trained_tf_lstm_model"]

    num_epochs = config["models"][model_name]["num_epochs_training"]
    window_size = config["models"][model_name]["window_size"]
    LSTM_units = config["models"][model_name]["LSTM_units"]
    Dense1_units = config["models"][model_name]["Dense1_units"]
    Dense2_units = config["models"][model_name]["Dense2_units"]
    Dense3_units = config["models"][model_name]["Dense3_units"]


### Pipeline nodes
    
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

def define_model() -> str:
    """Defines a model

    Returns:
        A model description in JSON
    """
    model = tf.keras.Sequential()
    model.add(LSTM(units=LSTM_units))
    model.add(Dense(Dense1_units, activation='relu'))
    model.add(Dense(Dense2_units))
    model.add(Dense(Dense3_units))

    loss = tf.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()
    metrics = [tf.metrics.MeanAbsoluteError()]

    model.compile(optimizer, loss, metrics=metrics)

    # TO BE READ FROM CONFIG
    model.build(input_shape)
    model.save(untrained_tf_lstm_model, overwrite=True)

    return model.to_json() # Kedro only saves serialisable objects


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
    # read model_description and choose model accordingly from a catalog of models
    # TODO

    model = tf.keras.saving.load_model(untrained_tf_lstm_model)
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=num_epochs)

    model.save(trained_tf_lstm_model, overwrite=True)

    return history


def evaluate_model(X_test: np.array, Y_test: np.array, history: History) -> float:
    """Evaluates the model

    Args:
        X_test: independent test data.
        Y_test: dependent test data.
        history: History object for training and validation.
    Returns:
        Scalar test loss.
    """
    model = tf.keras.saving.load_model(trained_tf_lstm_model)
    test_result = model.evaluate(X_test, Y_test)

    # TODO: these need to be setup according to model
    logger.info(f"""
        RESULTS. Overall Results of model training and evaluation for model {model.name}:

        \tTrain loss: {history.history["loss"][-1]}
        \tValidation loss: {history.history["val_loss"][-1]}
        \tTest loss: {test_result[0]}

        \tTrain mean absolute error: {history.history["mean_absolute_error"][-1]}
        \tValidation mean absolute error: {history.history["val_mean_absolute_error"][-1]}
        \tTest mean absolute error: {test_result[1]}

        \tHistory parameters: {history.params}
    """)

    return test_result