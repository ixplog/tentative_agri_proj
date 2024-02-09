from kedro.pipeline import Pipeline, node, pipeline
from .nodes import data_cleaning, split_data, compute_mean_std, normalise_data


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
                func=compute_mean_std,
                inputs=["train_logger_3_betula"],
                outputs=["train_mean", "train_std"],
                name="compute_mean_std_train_node",
            ),
            node(
                func=normalise_data,
                inputs=["train_logger_3_betula", "train_mean", "train_std"],
                outputs="train_logger_3_betula_norm",
                name="normalise_train_data_node",
            ),
            node(
                func=normalise_data,
                inputs=["val_logger_3_betula", "train_mean", "train_std"],
                outputs="val_logger_3_betula_norm",
                name="normalise_val_data_node",
            ),
            node(
                func=normalise_data,
                inputs=["test_logger_3_betula", "train_mean", "train_std"],
                outputs="test_logger_3_betula_norm",
                name="normalise_test_data_node",
            ),
        ]
    )
