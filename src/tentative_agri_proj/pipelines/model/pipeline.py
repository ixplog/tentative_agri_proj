from kedro.pipeline import Pipeline, node, pipeline
from .nodes import define_model, train_model, evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
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
                name="training_node",
            ),
            node(
                func=evaluate_model,
                inputs=["X_test", "Y_test", "history"],
                outputs="evaluation",
                name="eval_node",
            ),
        ]
    )
