import pandas as pd

from src.config import SHAP_SUMMARY_FILE
from src.logger import get_logger

logger = get_logger(__name__)


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
