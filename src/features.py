import numpy as np
import pandas as pd

from src.config import FEATURE_DATA_FILE, TARGET
from src.logger import get_logger
from src.utils import save_dataframe

logger = get_logger(__name__)


def _legal_ball_mask(deliveries: pd.DataFrame) -> pd.Series:
    if "extras_type" not in deliveries.columns:
        return pd.Series(True, index=deliveries.index)
    return ~deliveries["extras_type"].isin(["wides", "noballs"])


def _dismissal_mask(deliveries: pd.DataFrame) -> pd.Series:
    dismissed = deliveries["player_dismissed"].notna()
    if "dismissal_kind" not in deliveries.columns:
        return dismissed
    non_batter_dismissals = ["retired hurt", "retired out", "obstructing the field"]
    return dismissed & ~deliveries["dismissal_kind"].isin(non_batter_dismissals)


def build_second_innings_features(
    matches: pd.DataFrame, deliveries: pd.DataFrame, save: bool = True
) -> pd.DataFrame:
    logger.info("Building second-innings feature dataset")
    deliveries = deliveries.copy()
    matches = matches.copy()

    first_innings = deliveries[deliveries["inning"] == 1]
    targets = (
        first_innings.groupby("match_id")["total_runs"]
        .sum()
        .add(1)
        .rename("target_score")
        .reset_index()
    )

    second = deliveries[deliveries["inning"] == 2].copy()
    second["is_legal_ball"] = _legal_ball_mask(second).astype(int)
    second["is_wicket"] = _dismissal_mask(second).astype(int)

    second = second.sort_values(["match_id", "over", "ball"]).reset_index(drop=True)
    second["score"] = second.groupby("match_id")["total_runs"].cumsum()
    second["legal_balls_bowled"] = second.groupby("match_id")["is_legal_ball"].cumsum()
    second["wickets_lost"] = second.groupby("match_id")["is_wicket"].cumsum()

    second = second.merge(targets, on="match_id", how="inner")
    second = second.merge(
        matches[["id", "winner", "venue"]].rename(columns={"id": "match_id"}),
        on="match_id",
        how="left",
    )

    second["runs_left"] = second["target_score"] - second["score"]
    second["balls_left"] = 120 - second["legal_balls_bowled"]
    second["wickets_remaining"] = 10 - second["wickets_lost"]
    second["overs_completed"] = second["legal_balls_bowled"] / 6
    second["current_run_rate"] = np.where(
        second["legal_balls_bowled"] > 0,
        second["score"] * 6 / second["legal_balls_bowled"],
        0,
    )
    second["required_run_rate"] = np.where(
        second["balls_left"] > 0,
        second["runs_left"] * 6 / second["balls_left"],
        99,
    )
    second["run_rate_gap"] = second["required_run_rate"] - second["current_run_rate"]
    second["pressure_index"] = (
        second["required_run_rate"]
        * (11 - second["wickets_remaining"])
        / np.maximum(second["balls_left"] / 6, 1)
    )
    second["resource_pressure"] = second["runs_left"] / np.maximum(
        second["wickets_remaining"], 1
    )
    second["chase_progress"] = second["score"] / second["target_score"]
    second[TARGET] = (second["batting_team"] == second["winner"]).astype(int)

    feature_df = second[
        [
            "match_id",
            "batting_team",
            "bowling_team",
            "venue",
            "runs_left",
            "balls_left",
            "wickets_remaining",
            "current_run_rate",
            "required_run_rate",
            "target_score",
            "score",
            "overs_completed",
            "pressure_index",
            "resource_pressure",
            "chase_progress",
            "run_rate_gap",
            TARGET,
        ]
    ].copy()

    feature_df = feature_df[
        (feature_df["balls_left"] >= 0)
        & (feature_df["wickets_remaining"] >= 0)
        & (feature_df["runs_left"] > -30)
        & feature_df["venue"].notna()
    ]
    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    feature_df.dropna(inplace=True)

    if save:
        save_dataframe(feature_df, FEATURE_DATA_FILE)
        logger.info("Saved feature data to %s", FEATURE_DATA_FILE)
    return feature_df


def build_single_prediction_row(
    batting_team: str,
    bowling_team: str,
    target_score: int,
    score: int,
    overs_completed: float,
    wickets_lost: int,
    venue: str = "Unknown",
) -> pd.DataFrame:
    legal_balls_bowled = int(round(overs_completed * 6))
    balls_left = max(120 - legal_balls_bowled, 0)
    wickets_remaining = max(10 - wickets_lost, 0)
    runs_left = max(target_score - score, 0)
    current_run_rate = score * 6 / legal_balls_bowled if legal_balls_bowled else 0
    required_run_rate = runs_left * 6 / balls_left if balls_left else 99
    run_rate_gap = required_run_rate - current_run_rate
    pressure_index = required_run_rate * (11 - wickets_remaining) / max(balls_left / 6, 1)
    resource_pressure = runs_left / max(wickets_remaining, 1)
    chase_progress = score / target_score if target_score else 0

    return pd.DataFrame(
        [
            {
                "batting_team": batting_team,
                "bowling_team": bowling_team,
                "venue": venue,
                "runs_left": runs_left,
                "balls_left": balls_left,
                "wickets_remaining": wickets_remaining,
                "current_run_rate": current_run_rate,
                "required_run_rate": required_run_rate,
                "target_score": target_score,
                "score": score,
                "overs_completed": overs_completed,
                "pressure_index": pressure_index,
                "resource_pressure": resource_pressure,
                "chase_progress": chase_progress,
                "run_rate_gap": run_rate_gap,
            }
        ]
    )
