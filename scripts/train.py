import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_raw_data
from src.explainability import create_shap_summary
from src.features import build_second_innings_features
from src.model_trainer import train_and_select_model


def main() -> None:
    matches, deliveries = load_raw_data()
    feature_df = build_second_innings_features(matches, deliveries, save=True)
    best_model, metrics_df = train_and_select_model(feature_df)
    sample = feature_df.drop(columns=["batting_team_won", "match_id"]).sample(
        n=min(500, len(feature_df)), random_state=42
    )
    create_shap_summary(best_model, sample)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
