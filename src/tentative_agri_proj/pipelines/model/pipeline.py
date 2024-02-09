from kedro.pipeline import Pipeline, node, pipeline
from .nodes import get_windowed_data, define_model, train_model, evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_windowed_data,
                inputs=["train_logger_3_betula_norm"],
                outputs=["X_train", "Y_train"],
                name="get_windowed_train_data_node",
            ),
            node(
                func=get_windowed_data,
                inputs=["val_logger_3_betula_norm"],
                outputs=["X_val", "Y_val"],
                name="get_windowed_val_data_node",
            ),
            node(
                func=get_windowed_data,
                inputs=["test_logger_3_betula_norm"],
                outputs=["X_test", "Y_test"],
                name="get_windowed_test_data_node",
            ),
            node(
                func=define_model,
                inputs=None,
                outputs="model",
                name="define_model_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "Y_train", "X_val", "Y_val", "model"],
                outputs="history",
                name="training__model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["X_test", "Y_test", "history"],
                outputs="evaluation",
                name="eval_model_node",
            ),
        ]
    )
