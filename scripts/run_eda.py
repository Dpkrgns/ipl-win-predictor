import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import ARTIFACTS_DIR, TARGET
from src.data_loader import load_raw_data
from src.features import build_second_innings_features


def main() -> None:
    matches, deliveries = load_raw_data()
    df = build_second_innings_features(matches, deliveries, save=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=TARGET)
    plt.title("Second Innings Outcome Distribution")
    plt.xlabel("Batting team won")
    plt.ylabel("Ball states")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "class_balance.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="overs_completed", y="required_run_rate", hue=TARGET)
    plt.title("Required Run Rate Across Chase")
    plt.xlabel("Overs completed")
    plt.ylabel("Required run rate")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "required_run_rate_trend.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=TARGET, y="pressure_index")
    plt.title("Pressure Index by Match Result")
    plt.xlabel("Batting team won")
    plt.ylabel("Pressure index")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "pressure_index_boxplot.png", dpi=160)
    plt.close()

    numeric = [
        "runs_left",
        "balls_left",
        "wickets_remaining",
        "current_run_rate",
        "required_run_rate",
        "pressure_index",
        "run_rate_gap",
        TARGET,
    ]
    plt.figure(figsize=(9, 7))
    sns.heatmap(df[numeric].corr(), annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "correlation_heatmap.png", dpi=160)
    plt.close()

    print(f"Saved EDA charts to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
