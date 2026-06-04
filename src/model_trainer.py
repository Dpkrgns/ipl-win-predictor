import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from src.config import (
    FEATURE_IMPORTANCE_FILE,
    METRICS_FILE,
    MODEL_FILE,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)
from src.evaluation import evaluate_classifier, results_to_dataframe
from src.logger import get_logger
from src.preprocessing import build_preprocessor
from src.utils import save_dataframe, save_object

logger = get_logger(__name__)


def _candidate_models() -> dict:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=8,
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )
    except ImportError:
        warnings.warn("xgboost is not installed; skipping XGBoost.")

    try:
        from lightgbm import LGBMClassifier

        models["LightGBM"] = LGBMClassifier(
            n_estimators=350,
            learning_rate=0.04,
            num_leaves=31,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            verbose=-1,
        )
    except ImportError:
        warnings.warn("lightgbm is not installed; skipping LightGBM.")

    return models


def _match_level_split(df: pd.DataFrame):
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    x = df.drop(columns=[TARGET])
    y = df[TARGET]
    groups = df["match_id"]
    train_idx, test_idx = next(splitter.split(x, y, groups=groups))
    return x.iloc[train_idx], x.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def train_and_select_model(feature_df: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame]:
    logger.info("Training candidate models")
    x_train, x_test, y_train, y_test = _match_level_split(feature_df)
    x_train = x_train.drop(columns=["match_id"])
    x_test = x_test.drop(columns=["match_id"])

    results = []
    trained = {}
    for model_name, estimator in _candidate_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )
        logger.info("Fitting %s", model_name)
        pipeline.fit(x_train, y_train)
        results.append(evaluate_classifier(model_name, pipeline, x_test, y_test))
        trained[model_name] = pipeline

    metrics_df = results_to_dataframe(results)
    best_model_name = metrics_df.iloc[0]["model_name"]
    best_model = trained[best_model_name]

    save_object(best_model, MODEL_FILE)
    save_dataframe(metrics_df, METRICS_FILE)
    save_feature_importance(best_model, FEATURE_IMPORTANCE_FILE)

    logger.info("Selected %s as final model", best_model_name)
    return best_model, metrics_df


def save_feature_importance(model: Pipeline, path) -> None:
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importance = abs(estimator.coef_[0])
    else:
        return

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .head(30)
    )
    save_dataframe(importance_df, path)
