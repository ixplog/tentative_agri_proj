from kedro.pipeline import Pipeline, node, pipeline
from .nodes import data_cleaning, split_data, normalise_data, define_model, windowing, train_model, evaluate_model, save_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=data_cleaning,
                inputs=["logger_3_betula"],
                outputs="logger_3_betula_cleaned",
                name="data_cleaning_node",
            ),
            node(
                func=split_data,
                inputs=["logger_3_betula_cleaned"],
                outputs=["train_logger_3_betula", "val_logger_3_betula", "test_logger_3_betula"],
                name="split_data_node",
            ),
            node(
                func=normalise_data,
                inputs=["train_logger_3_betula", "val_logger_3_betula", "test_logger_3_betula"],
                outputs=["train_logger_3_betula_norm", "val_logger_3_betula_norm", "test_logger_3_betula_norm"],
                name="normalise_data_node",
            ),
            node(
                func=define_model,
                inputs=None,
                outputs="model",
                name="define_model_node",
            ),
            node(
                func=windowing,
                inputs=["train_logger_3_betula_norm", 10, ["RH.percent", "Tair.C"]],
                outputs=["X_train", "Y_train"],
                name="windowing_train_node",
            ),
            node(
                func=windowing,
                inputs=["val_logger_3_betula_norm", 10, ["RH.percent", "Tair.C"]],
                outputs=["X_val", "Y_val"],
                name="windowing_val_node",
            ),
            node(
                func=windowing,
                inputs=["test_logger_3_betula_norm", 10, ["RH.percent", "Tair.C"]],
                outputs=["X_test", "Y_test"],
                name="windowing_test_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "Y_train", "X_val", "Y_val", "model"],
                outputs=["trained_model", "history"],
                name="training_node",
            ),
            node(
                func=evaluate_model,
                inputs=["X_test", "Y_test", "trained_model"],
                outputs=["evaluation"],
                name="eval_node",
            ),
            node(
                func=save_model,
                inputs=["trained_model", "tf_lstm_model"],
                outputs=[],
                name="save_model_node",
            ),
        ]
    )