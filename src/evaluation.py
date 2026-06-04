from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvaluationResult:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


def evaluate_classifier(model_name: str, model, x_test, y_test) -> EvaluationResult:
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]
    return EvaluationResult(
        model_name=model_name,
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, zero_division=0),
        recall=recall_score(y_test, y_pred, zero_division=0),
        f1=f1_score(y_test, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_test, y_prob),
    )


def results_to_dataframe(results: list[EvaluationResult]) -> pd.DataFrame:
    return pd.DataFrame([result.__dict__ for result in results]).sort_values(
        by="roc_auc", ascending=False
    )
