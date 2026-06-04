from src.config import MODEL_FILE
from src.exceptions import ModelArtifactError
from src.features import build_single_prediction_row
from src.utils import load_object


class IPLWinPredictor:
    def __init__(self, model_path=MODEL_FILE):
        try:
            self.model = load_object(model_path)
        except FileNotFoundError as exc:
            raise ModelArtifactError(
                "Model artifact is missing. Run `python scripts/train.py` after adding data."
            ) from exc

    def predict_probability(
        self,
        batting_team: str,
        bowling_team: str,
        target_score: int,
        score: int,
        overs_completed: float,
        wickets_lost: int,
        venue: str = "Unknown",
    ) -> dict:
        row = build_single_prediction_row(
            batting_team=batting_team,
            bowling_team=bowling_team,
            target_score=target_score,
            score=score,
            overs_completed=overs_completed,
            wickets_lost=wickets_lost,
            venue=venue,
        )
        features = row.iloc[0].to_dict()

        if score >= target_score:
            return {
                "win_probability": 100.0,
                "lose_probability": 0.0,
                "features": features,
                "decision_note": "Target has already been chased.",
            }

        if wickets_lost >= 10:
            return {
                "win_probability": 0.0,
                "lose_probability": 100.0,
                "features": features,
                "decision_note": "All wickets are down before reaching the target.",
            }

        if features["balls_left"] <= 0:
            return {
                "win_probability": 0.0,
                "lose_probability": 100.0,
                "features": features,
                "decision_note": "No legal balls are left before reaching the target.",
            }

        win_probability = float(self.model.predict_proba(row)[0, 1])
        return {
            "win_probability": round(win_probability * 100, 2),
            "lose_probability": round((1 - win_probability) * 100, 2),
            "features": features,
            "decision_note": "Model prediction for a live chase state.",
        }
