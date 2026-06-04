from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
LOG_DIR = ROOT_DIR / "logs"

MATCHES_FILE = RAW_DATA_DIR / "matches.csv"
DELIVERIES_FILE = RAW_DATA_DIR / "deliveries.csv"
FEATURE_DATA_FILE = PROCESSED_DATA_DIR / "second_innings_features.csv"

MODEL_FILE = ARTIFACTS_DIR / "ipl_win_pipeline.joblib"
METRICS_FILE = ARTIFACTS_DIR / "metrics.csv"
FEATURE_IMPORTANCE_FILE = ARTIFACTS_DIR / "feature_importance.csv"
SHAP_SUMMARY_FILE = ARTIFACTS_DIR / "shap_summary.png"

RANDOM_STATE = 42
TEST_SIZE = 0.2

TEAM_NAME_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
}

CURRENT_IPL_TEAMS = [
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bangalore",
    "Sunrisers Hyderabad",
]

CATEGORICAL_FEATURES = ["batting_team", "bowling_team", "venue"]
NUMERIC_FEATURES = [
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
]
TARGET = "batting_team_won"
