from kedro.pipeline import Pipeline, node, pipeline
from .nodes import data_cleaning, split_data, normalise_data, get_windowed_data


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
                func=get_windowed_data,
                inputs=["train_logger_3_betula_norm"],
                outputs=["X_train", "Y_train"],
                name="windowing_train_node",
            ),
            node(
                func=get_windowed_data,
                inputs=["val_logger_3_betula_norm"],
                outputs=["X_val", "Y_val"],
                name="windowing_val_node",
            ),
            node(
                func=get_windowed_data,
                inputs=["test_logger_3_betula_norm"],
                outputs=["X_test", "Y_test"],
                name="windowing_test_node",
            ),
        ]
    )
