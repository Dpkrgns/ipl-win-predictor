from src.config import MODEL_FILE
from src.exceptions import ModelArtifactError
from src.explainability import explain_prediction
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
        include_explanation: bool = True,
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
            return self._finalize(
                {
                "win_probability": 100.0,
                "lose_probability": 0.0,
                "features": features,
                "decision_note": "Target has already been chased.",
                },
                row,
                features,
                include_explanation,
            )

        if wickets_lost >= 10:
            return self._finalize(
                {
                "win_probability": 0.0,
                "lose_probability": 100.0,
                "features": features,
                "decision_note": "All wickets are down before reaching the target.",
                },
                row,
                features,
                include_explanation,
            )

        if features["balls_left"] <= 0:
            return self._finalize(
                {
                "win_probability": 0.0,
                "lose_probability": 100.0,
                "features": features,
                "decision_note": "No legal balls are left before reaching the target.",
                },
                row,
                features,
                include_explanation,
            )

        win_probability = float(self.model.predict_proba(row)[0, 1])
        return self._finalize(
            {
                "win_probability": round(win_probability * 100, 2),
                "lose_probability": round((1 - win_probability) * 100, 2),
                "features": features,
                "decision_note": "Model prediction for a live chase state.",
            },
            row,
            features,
            include_explanation,
        )

    def _finalize(self, result: dict, row, features: dict, include_explanation: bool) -> dict:
        result = self._result(result, features)
        result["explanations"] = explain_prediction(self.model, row) if include_explanation else []
        return result

    def build_probability_timeline(
        self,
        batting_team: str,
        bowling_team: str,
        target_score: int,
        score: int,
        overs_completed: float,
        wickets_lost: int,
        venue: str = "Unknown",
    ) -> list[dict]:
        """Build a projected probability curve using the current chase pace."""
        legal_balls = int(round(overs_completed * 6))
        current_rate = score * 6 / legal_balls if legal_balls else target_score / 20
        remaining_balls = max(120 - legal_balls, 1)
        required_rate = max(target_score - score, 0) * 6 / remaining_balls
        checkpoints = list(range(0, 21, 2))
        if overs_completed not in checkpoints and 0 < overs_completed < 20:
            checkpoints.append(overs_completed)
        checkpoints = sorted(set(checkpoints))
        timeline = []

        for checkpoint in checkpoints:
            checkpoint_balls = int(round(checkpoint * 6))
            if checkpoint_balls <= legal_balls:
                checkpoint_score = min(target_score, round(current_rate * checkpoint_balls / 6))
                checkpoint_wickets = round(wickets_lost * checkpoint_balls / max(legal_balls, 1))
            else:
                future_balls = checkpoint_balls - legal_balls
                checkpoint_score = min(
                    target_score,
                    score + round(required_rate * future_balls / 6),
                )
                checkpoint_wickets = wickets_lost

            point = self.predict_probability(
                batting_team=batting_team,
                bowling_team=bowling_team,
                target_score=target_score,
                score=checkpoint_score,
                overs_completed=checkpoint,
                wickets_lost=min(checkpoint_wickets, 10),
                venue=venue,
                include_explanation=False,
            )
            timeline.append(
                {
                    "over": checkpoint,
                    "probability": point["win_probability"],
                    "is_current": abs(checkpoint - overs_completed) < 0.01,
                }
            )
        return timeline

    @staticmethod
    def _result(result: dict, features: dict) -> dict:
        required_rate = float(features["required_run_rate"])
        current_rate = float(features["current_run_rate"])
        wickets_remaining = int(features["wickets_remaining"])
        rate_gap = float(features["run_rate_gap"])

        if result["win_probability"] >= 70:
            outlook = "Chase is in control"
        elif result["win_probability"] >= 45:
            outlook = "A contest in the balance"
        else:
            outlook = "Defending side has the edge"

        if wickets_remaining <= 3:
            key_signal = "Wickets are the scarce resource"
        elif rate_gap > 4:
            key_signal = "Required rate is creating pressure"
        elif rate_gap < 0:
            key_signal = "Current scoring is ahead of the ask"
        else:
            key_signal = "The chase is tracking close to par"

        result["analysis"] = {
            "outlook": outlook,
            "key_signal": key_signal,
            "current_rate": round(current_rate, 2),
            "required_rate": round(required_rate, 2),
            "wickets_remaining": wickets_remaining,
        }
        return result
