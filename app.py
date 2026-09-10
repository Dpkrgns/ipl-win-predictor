from flask import Flask, jsonify, render_template, request
import pandas as pd

from src.config import CURRENT_IPL_TEAMS, FEATURE_IMPORTANCE_FILE, METRICS_FILE, MODEL_FILE
from src.exceptions import ModelArtifactError
from src.predictor import IPLWinPredictor

app = Flask(__name__)

DEFAULT_VENUES = [
    "Wankhede Stadium",
    "M Chinnaswamy Stadium",
    "Eden Gardens",
    "Arun Jaitley Stadium",
    "MA Chidambaram Stadium",
    "Rajiv Gandhi International Stadium",
    "Sawai Mansingh Stadium",
    "Narendra Modi Stadium",
    "Unknown",
]

DEFAULT_FORM_DATA = {
    "batting_team": "Chennai Super Kings",
    "bowling_team": "Delhi Capitals",
    "venue": "Wankhede Stadium",
    "target_score": "178",
    "score": "92",
    "overs_completed": "11.4",
    "wickets_lost": "3",
}


def _validate_form(form):
    batting_team = form.get("batting_team")
    bowling_team = form.get("bowling_team")
    if batting_team == bowling_team:
        raise ValueError("Batting and bowling teams must be different.")

    target = int(form.get("target_score", 0))
    score = int(form.get("score", 0))
    overs = _parse_cricket_overs(form.get("overs_completed", "0"))
    wickets = int(form.get("wickets_lost", 0))

    if target <= 0:
        raise ValueError("Target must be positive.")
    if score < 0 or score > target + 30:
        raise ValueError("Current score looks invalid for this target.")
    if overs < 0 or overs > 20:
        raise ValueError("Overs completed must be between 0 and 20.")
    if wickets < 0 or wickets > 10:
        raise ValueError("Wickets lost must be between 0 and 10.")

    return batting_team, bowling_team, target, score, overs, wickets


def _parse_cricket_overs(value: str) -> float:
    text = str(value).strip()
    if not text:
        raise ValueError("Overs completed is required.")
    if "." not in text:
        overs = int(text)
        balls = 0
    else:
        over_text, ball_text = text.split(".", 1)
        overs = int(over_text or 0)
        balls = int(ball_text or 0)
    if balls > 5:
        raise ValueError("Use cricket over notation: 11.4 means 11 overs and 4 balls.")
    legal_balls = overs * 6 + balls
    return legal_balls / 6


def _legal_balls_from_overs(value: str) -> int:
    return round(_parse_cricket_overs(value) * 6)


def _load_records(path):
    if not path.exists():
        return []
    try:
        records = pd.read_csv(path).replace({pd.NA: None}).to_dict(orient="records")
        for record in records:
            if "f1_score" not in record and "f1" in record:
                record["f1_score"] = record["f1"]
        return records
    except (OSError, ValueError, pd.errors.ParserError):
        return []


def _predict_from_form(form, include_explanation=True):
    batting_team, bowling_team, target, score, overs, wickets = _validate_form(form)
    predictor = IPLWinPredictor()
    venue = form.get("venue", "Unknown")
    result = predictor.predict_probability(
        batting_team=batting_team,
        bowling_team=bowling_team,
        target_score=target,
        score=score,
        overs_completed=overs,
        wickets_lost=wickets,
        venue=venue,
        include_explanation=include_explanation,
    )
    result["timeline"] = predictor.build_probability_timeline(
        batting_team=batting_team,
        bowling_team=bowling_team,
        target_score=target,
        score=score,
        overs_completed=overs,
        wickets_lost=wickets,
        venue=venue,
    )
    scenario_inputs = [
        ("Current state", score, wickets, "Baseline"),
        ("12 runs added", min(score + 12, target), wickets, "If the next phase scores 12"),
        ("One wicket lost", score, min(wickets + 1, 10), "If the next wicket falls"),
    ]
    result["scenarios"] = []
    for label, scenario_score, scenario_wickets, caption in scenario_inputs:
        scenario = predictor.predict_probability(
            batting_team=batting_team,
            bowling_team=bowling_team,
            target_score=target,
            score=scenario_score,
            overs_completed=overs,
            wickets_lost=scenario_wickets,
            venue=venue,
            include_explanation=False,
        )
        result["scenarios"].append(
            {"label": label, "caption": caption, "probability": scenario["win_probability"]}
        )
    return result


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    scenario_results = []
    error = None
    form_data = DEFAULT_FORM_DATA.copy()
    simulator_balls = _legal_balls_from_overs(form_data["overs_completed"])
    if request.method == "POST":
        form_data.update(request.form.to_dict())
        try:
            simulator_balls = _legal_balls_from_overs(form_data["overs_completed"])
        except (TypeError, ValueError):
            simulator_balls = _legal_balls_from_overs(DEFAULT_FORM_DATA["overs_completed"])
        try:
            result = _predict_from_form(request.form)
            scenario_results = result["scenarios"]
        except (ValueError, ModelArtifactError) as exc:
            error = str(exc)

    return render_template(
        "index.html",
        teams=CURRENT_IPL_TEAMS,
        venues=DEFAULT_VENUES,
        form_data=form_data,
        result=result,
        scenario_results=scenario_results,
        metrics=_load_records(METRICS_FILE),
        feature_importance=_load_records(FEATURE_IMPORTANCE_FILE),
        model_ready=MODEL_FILE.exists(),
        simulator_balls=simulator_balls,
        error=error,
    )


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "model_ready": MODEL_FILE.exists()})


@app.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True) or {}
    try:
        result = _predict_from_form(payload)
        return jsonify(result)
    except (ValueError, ModelArtifactError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/evaluation")
def api_evaluation():
    return jsonify(
        {
            "model_ready": MODEL_FILE.exists(),
            "metrics": _load_records(METRICS_FILE),
            "feature_importance": _load_records(FEATURE_IMPORTANCE_FILE),
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
