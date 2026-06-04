import pandas as pd

from src.config import DELIVERIES_FILE, MATCHES_FILE, TEAM_NAME_MAP
from src.exceptions import DataValidationError
from src.logger import get_logger

logger = get_logger(__name__)


REQUIRED_MATCH_COLUMNS = {"id", "winner", "venue"}
REQUIRED_DELIVERY_COLUMNS = {
    "match_id",
    "inning",
    "batting_team",
    "bowling_team",
    "over",
    "ball",
    "total_runs",
    "batsman_runs",
    "player_dismissed",
}


def normalize_team_names(df: pd.DataFrame) -> pd.DataFrame:
    team_cols = ["team1", "team2", "toss_winner", "winner", "batting_team", "bowling_team"]
    for col in team_cols:
        if col in df.columns:
            df[col] = df[col].replace(TEAM_NAME_MAP)
    return df


def load_raw_data(
    matches_path=MATCHES_FILE, deliveries_path=DELIVERIES_FILE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Loading raw IPL data")
    if not matches_path.exists() or not deliveries_path.exists():
        raise FileNotFoundError(
            "Place matches.csv and deliveries.csv inside data/raw before training."
        )

    matches = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)
    validate_raw_data(matches, deliveries)
    return normalize_team_names(matches), normalize_team_names(deliveries)


def validate_raw_data(matches: pd.DataFrame, deliveries: pd.DataFrame) -> None:
    missing_matches = REQUIRED_MATCH_COLUMNS - set(matches.columns)
    missing_deliveries = REQUIRED_DELIVERY_COLUMNS - set(deliveries.columns)
    if missing_matches:
        raise DataValidationError(f"matches.csv missing columns: {sorted(missing_matches)}")
    if missing_deliveries:
        raise DataValidationError(
            f"deliveries.csv missing columns: {sorted(missing_deliveries)}"
        )
