# IPL Match Win Predictor

An end-to-end machine learning project that predicts the probability of a batting team winning an IPL match during the second innings. The project uses IPL ball-by-ball data, cricket-specific feature engineering, model comparison, explainability, and a Flask web application.

## Key Highlights

- Predicts live win probability between 0 and 100 percent.
- Uses second-innings ball-by-ball IPL data.
- Engineers cricket-specific features such as runs left, balls left, wickets remaining, current run rate, required run rate, pressure index, chase progress, resource pressure, and run-rate gap.
- Compares Logistic Regression, Random Forest, XGBoost, and optional LightGBM.
- Evaluates models using Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
- Uses match-level train-test splitting to reduce leakage from ball-by-ball records.
- Includes feature importance and SHAP-based explainability when dependencies are available.
- Deploys the final model with a clean Flask interface.
- Handles cricket terminal states such as all wickets lost, target reached, and no balls remaining.

## Problem Statement

Given a live second-innings IPL chase state, predict:

```text
P(second-innings batting team wins | current match state)
```

The output is:

- Win probability
- Lose probability
- Visual probability bar
- Important engineered chase features

## Dataset

Use IPL ball-by-ball data with two CSV files:

- `matches.csv`: match metadata such as match id, winner, venue, teams, and toss information.
- `deliveries.csv`: ball-by-ball details such as innings, batting team, bowling team, runs, extras, overs, balls, and wickets.

Place both files here:

```text
data/raw/matches.csv
data/raw/deliveries.csv
```

These raw CSV files are intentionally ignored by Git because datasets can be large and may have redistribution restrictions.

## Target Variable

For every second-innings ball state:

```python
batting_team_won = 1 if second_innings_batting_team == match_winner else 0
```

The final result is used only as the label. Features are computed from information available at the current ball, which helps avoid future-information leakage.

## Feature Engineering

| Feature | Why it matters |
|---|---|
| `runs_left` | Measures remaining chase demand. |
| `balls_left` | Measures time/resources left in the innings. |
| `wickets_remaining` | Captures batting resources and risk capacity. |
| `current_run_rate` | Shows scoring pace so far. |
| `required_run_rate` | Shows future scoring pressure. |
| `run_rate_gap` | Compares required rate with current scoring speed. |
| `pressure_index` | Combines required run rate, wickets lost, and overs remaining. |
| `resource_pressure` | Measures runs left per wicket remaining. |
| `chase_progress` | Measures how much of the target has already been achieved. |
| `target_score` | Adds match context for chase difficulty. |
| `venue` | Captures ground-specific scoring/chasing patterns. |
| `batting_team`, `bowling_team` | Captures team-level historical behavior. |

## ML Workflow

1. Load `matches.csv` and `deliveries.csv`.
2. Clean team names and validate required columns.
3. Build second-innings ball-state features.
4. Split train/test data by `match_id` to reduce leakage.
5. Preprocess numeric and categorical features.
6. Train multiple candidate models.
7. Evaluate using classification and ranking metrics.
8. Save the best model pipeline.
9. Generate feature importance and SHAP summary when possible.
10. Serve predictions through Flask.

## Models Compared

- Logistic Regression: simple, fast, and interpretable baseline.
- Random Forest: handles nonlinear feature interactions.
- XGBoost: strong structured-data model for tabular ML.
- LightGBM: optional fast gradient boosting model.

The final model is selected using ROC-AUC because this is a probability-ranking problem.

## Evaluation Metrics

- Accuracy: overall correct win/loss classification rate.
- Precision: among predicted winning states, how many actually became wins.
- Recall: among actual winning chase states, how many the model identified.
- F1 Score: balance between precision and recall.
- ROC-AUC: how well the model ranks winning states above losing states across thresholds.

## Project Structure

```text
ipl-win-predictor/
├── artifacts/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   └── interview_prep.md
├── notebooks/
├── scripts/
│   ├── run_eda.py
│   └── train.py
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── exceptions.py
│   ├── explainability.py
│   ├── features.py
│   ├── logger.py
│   ├── model_trainer.py
│   ├── predictor.py
│   ├── preprocessing.py
│   └── utils.py
├── static/
├── templates/
├── app.py
├── requirements.txt
└── README.md
```

## How To Run

Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Add dataset files:

```text
data/raw/matches.csv
data/raw/deliveries.csv
```

Run EDA:

```powershell
python scripts/run_eda.py
```

Train the model:

```powershell
python scripts/train.py
```

Start the Flask app:

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:5000/
```

## Flask Inputs

- Batting team
- Bowling team
- Venue
- Target
- Current score
- Overs completed
- Wickets lost

Use cricket over notation: `11.4` means 11 overs and 4 legal balls.

## Explainability

The project saves feature importance to:

```text
artifacts/feature_importance.csv
```

If SHAP is installed and supported for the selected model, it also saves:

```text
artifacts/shap_summary.png
```

## Future Improvements

- Add batter and bowler form features.
- Add venue par score and chasing history.
- Add toss, dew, season, and playoff-pressure indicators.
- Add probability calibration and reliability curves.
- Use season-wise time-based validation.
- Add unit tests for feature calculations.

## Interview Preparation

See:

```text
docs/interview_prep.md
```
