import pandas as pd

from src.config import SHAP_SUMMARY_FILE
from src.logger import get_logger

logger = get_logger(__name__)


FEATURE_LABELS = {
    "required_run_rate": "Required Run Rate",
    "wickets_remaining": "Wickets Remaining",
    "current_run_rate": "Current Run Rate",
    "chase_progress": "Chase Progress",
    "venue": "Venue",
    "runs_left": "Runs Left",
    "balls_left": "Balls Left",
    "pressure_index": "Pressure Index",
    "resource_pressure": "Resource Pressure",
    "target_score": "Target Score",
    "score": "Current Score",
    "overs_completed": "Overs Completed",
    "run_rate_gap": "Run Rate Gap",
    "batting_team": "Batting Team",
    "bowling_team": "Bowling Team",
}


def explain_prediction(model, sample: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """Return signed local SHAP contributions for a single prediction row."""
    try:
        import shap

        estimator = model.named_steps["model"]
        preprocessor = model.named_steps["preprocessor"]
        transformed = preprocessor.transform(sample)
        feature_names = preprocessor.get_feature_names_out()
        explanation = shap.Explainer(estimator, transformed)(transformed)
        values = explanation.values[0]
        if values.ndim > 1:
            values = values[:, -1]

        grouped: dict[str, float] = {}
        for name, value in zip(feature_names, values):
            raw_name = str(name).split("__", 1)[-1]
            base_name = next(
                (feature for feature in FEATURE_LABELS if raw_name.startswith(feature)),
                raw_name,
            )
            grouped[base_name] = grouped.get(base_name, 0.0) + float(value)

        ranked = sorted(grouped.items(), key=lambda item: abs(item[1]), reverse=True)[:top_n]
        max_value = max((abs(value) for _, value in ranked), default=1.0)
        return [
            {
                "label": FEATURE_LABELS.get(name, name.replace("_", " ").title()),
                "value": round(abs(value), 3),
            "strength": round(abs(value) / max_value * 100),
                "direction": "supports" if value >= 0 else "against",
                "signed_value": round(value, 3),
            }
            for name, value in ranked
            if abs(value) > 0.0001
        ]
    except Exception as exc:
        logger.warning("Could not explain prediction with SHAP: %s", exc)
        return []


def create_shap_summary(model, sample: pd.DataFrame, output_path=SHAP_SUMMARY_FILE) -> bool:
    try:
        import matplotlib.pyplot as plt
        import shap
    except ImportError:
        logger.warning("SHAP or matplotlib is not installed; skipping SHAP summary.")
        return False

    estimator = model.named_steps["model"]
    preprocessor = model.named_steps["preprocessor"]
    transformed = preprocessor.transform(sample)
    feature_names = preprocessor.get_feature_names_out()

    try:
        explainer = shap.Explainer(estimator, transformed)
        shap_values = explainer(transformed)
        shap.summary_plot(shap_values, features=transformed, feature_names=feature_names, show=False)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close()
        return True
    except Exception as exc:
        logger.warning("Could not create SHAP summary: %s", exc)
        return False
