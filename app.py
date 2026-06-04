from flask import Flask, render_template, request

from src.config import CURRENT_IPL_TEAMS
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


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    form_data = DEFAULT_FORM_DATA.copy()
    if request.method == "POST":
        form_data.update(request.form.to_dict())
        try:
            batting_team, bowling_team, target, score, overs, wickets = _validate_form(
                request.form
            )
            predictor = IPLWinPredictor()
            result = predictor.predict_probability(
                batting_team=batting_team,
                bowling_team=bowling_team,
                target_score=target,
                score=score,
                overs_completed=overs,
                wickets_lost=wickets,
                venue=request.form.get("venue", "Unknown"),
            )
        except (ValueError, ModelArtifactError) as exc:
            error = str(exc)

    return render_template(
        "index.html",
        teams=CURRENT_IPL_TEAMS,
        venues=DEFAULT_VENUES,
        form_data=form_data,
        result=result,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
