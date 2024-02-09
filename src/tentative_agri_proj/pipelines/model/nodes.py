import numpy as np
import yaml

import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import History

# Paramaters read

with open('conf/base/parameters.yml') as params:
    config = yaml.safe_load(params)

    untrained_tf_lstm_model = config["untrained_tf_lstm_model"]
    trained_tf_lstm_model = config["trained_tf_lstm_model"]

    num_epochs = config["num_epochs_training"]

# Pipeline nodes

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


def evaluate_model(X_test: np.array, Y_test: np.array, history) -> float:
    """Evaluates the model

    Args:
        X_test: independent test data.
        Y_test: dependent test data.
        model: model to be evaluated.
    Returns:
        Scalar test loss.
    """
    model = tf.keras.saving.load_model(trained_tf_lstm_model)
    test_result = model.evaluate(X_test, Y_test)

    # TODO: compare with history

    print(f"Test evaluation: {test_result}")

    return test_result