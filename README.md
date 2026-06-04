# IPL Match Win Predictor

Predict the probability of the batting team winning an IPL match during the second innings. The output is a win probability from 0 to 100 percent and a loss probability for the same chase state.

This project is built for an internship interview discussion: it focuses on feature engineering, clean evaluation, model comparison, explainability, and a simple Flask deployment.

## Problem Statement

Given the live state of an IPL chase, predict:

```text
P(second-innings batting team wins | current match state)
```

The model is trained on ball-by-ball second-innings states. If the batting team eventually wins the match, the target is 1. Otherwise, the target is 0.

## Dataset

Use IPL ball-by-ball data with two CSV files:

- `data/raw/matches.csv`: match metadata such as match id, winner, venue, teams, and toss information.
- `data/raw/deliveries.csv`: every delivery in every match, including inning, batting team, bowling team, over, ball, runs, extras, wickets, and dismissal type.

A common source is the Kaggle IPL dataset containing `matches.csv` and `deliveries.csv`. Put both files in `data/raw/`.

The project expects these columns:

- From `matches.csv`: `id`, `winner`, `venue`
- From `deliveries.csv`: `match_id`, `inning`, `batting_team`, `bowling_team`, `over`, `ball`, `total_runs`, `batsman_runs`, `player_dismissed`
- Optional but useful: `extras_type`, `dismissal_kind`

## Target Variable

For every second-innings ball state:

```python
batting_team_won = 1 if second_innings_batting_team == match_winner else 0
```

This is not leakage because the final result is used only as the label. The features are built only from information available up to the current ball.

## Data Cleaning and Preprocessing

Implemented in `src/data_loader.py` and `src/features.py`.

- Standardizes old team names, for example Delhi Daredevils to Delhi Capitals.
- Uses only second innings for prediction rows.
- Computes target score from first innings total plus one.
- Counts only legal deliveries for balls left when `extras_type` is available.
- Excludes non-standard dismissal noise such as retired hurt when possible.
- Drops invalid rows with missing venue, impossible balls left, impossible wickets, or infinite run-rate values.
- Uses median imputation and scaling for numeric features.
- Uses one-hot encoding for categorical features.

## Feature Engineering

| Feature | Source | Why it helps | Expected impact |
|---|---|---|---|
| `runs_left` | `target_score - current_score` | Measures remaining chase demand. | Higher runs left usually lowers win probability. |
| `balls_left` | Legal balls remaining from 120. | Captures time/resource availability. | More balls left usually increases win probability. |
| `wickets_remaining` | 10 minus cumulative wickets. | Captures batting resources and risk capacity. | More wickets usually increases win probability. |
| `current_run_rate` | Current score per over. | Shows scoring momentum so far. | Higher current rate often increases win probability. |
| `required_run_rate` | Runs left per over remaining. | Shows future scoring pressure. | Higher required rate usually lowers win probability. |
| `target_score` | First innings score + 1. | Contextualizes whether the chase is small or large. | Larger targets are harder, but effect depends on current state. |
| `overs_completed` | Legal balls bowled / 6. | Helps model phase of innings. | Late innings magnify each run and wicket. |
| `run_rate_gap` | Required RR minus current RR. | Directly compares demand versus scoring speed. | Larger gap usually lowers win probability. |
| `pressure_index` | Required RR, wickets lost, overs left. | Cricket-specific interaction feature for chase stress. | Higher pressure usually lowers win probability. |
| `resource_pressure` | Runs left per wicket remaining. | Measures how much work each remaining wicket must support. | Higher value usually lowers win probability. |
| `chase_progress` | Current score divided by target. | Shows how much of the chase has already been completed. | Higher value usually increases win probability. |
| `venue` | Match metadata. | Some grounds are easier for chasing or high scoring. | Direction depends on venue history. |
| `batting_team`, `bowling_team` | Ball data. | Captures team-level historical strength. | Direction depends on team matchup. |

## Exploratory Data Analysis

Run:

```bash
python scripts/run_eda.py
```

It creates charts in `artifacts/`:

- `class_balance.png`: whether winning and losing chase states are balanced.
- `required_run_rate_trend.png`: how required rate changes across overs for wins and losses.
- `pressure_index_boxplot.png`: whether pressure is higher in failed chases.
- `correlation_heatmap.png`: relationships among numeric features.

Expected trends to discuss:

- Winning chases usually keep required run rate controlled.
- Losing chases often show a widening run-rate gap after the middle overs.
- Wickets remaining matters more late in the innings because there is less recovery time.
- Ball states are not independent; one match contributes many rows, so evaluation must split by match.

## Machine Learning Pipeline

The training script performs:

1. Load raw data.
2. Validate required columns.
3. Clean team names and match states.
4. Build second-innings features.
5. Split train and test by `match_id`.
6. Preprocess numeric and categorical columns.
7. Train multiple models.
8. Evaluate metrics.
9. Save final model and reports.

Run:

```bash
python scripts/train.py
```

Saved artifacts:

- `artifacts/ipl_win_pipeline.joblib`
- `artifacts/metrics.csv`
- `artifacts/feature_importance.csv`
- `artifacts/shap_summary.png` when SHAP succeeds

## Models Compared

### Logistic Regression

Advantages:

- Interpretable baseline.
- Fast to train.
- Useful for explaining feature direction.

Disadvantages:

- Struggles with nonlinear interactions unless engineered manually.
- May underfit late-over pressure patterns.

### Random Forest

Advantages:

- Captures nonlinear feature interactions.
- Robust to noisy tabular data.
- Provides feature importance.

Disadvantages:

- Probabilities can be less calibrated.
- Larger and less directly interpretable than logistic regression.

### XGBoost

Advantages:

- Strong performance on structured tabular data.
- Captures complex interactions.
- Often high ROC-AUC with careful regularization.

Disadvantages:

- More hyperparameters.
- Needs explainability tools for clear interpretation.

### LightGBM

Advantages:

- Fast gradient boosting.
- Strong for larger tabular datasets.

Disadvantages:

- Optional dependency.
- Can overfit if not validated carefully.

The final model is selected by highest ROC-AUC, because this is a probability-ranking problem. In an interview, also discuss calibration if the exact probability value is business-critical.

## Evaluation Metrics

- Accuracy: Overall fraction of correct win/loss classifications.
- Precision: When the model predicts a likely chase win, how often was it correct?
- Recall: Among actual successful chases, how many did the model identify?
- F1 Score: Balance between precision and recall.
- ROC-AUC: How well the model ranks winning chase states above losing chase states across thresholds.

Important: this project uses `GroupShuffleSplit` by match id. That prevents balls from the same match appearing in both train and test.

## Explainability

The project includes:

- Global feature importance saved to `artifacts/feature_importance.csv`.
- SHAP summary plot saved to `artifacts/shap_summary.png` when supported by the installed model stack.

How to explain a prediction:

- High `required_run_rate` and high `pressure_index` usually push probability down.
- High `wickets_remaining` and many `balls_left` usually push probability up.
- Team and venue features capture historical context, but they should not be overinterpreted as causal.

## Flask Deployment

Run:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

User inputs:

- Batting Team
- Bowling Team
- Venue
- Target
- Current Score
- Overs Completed
- Wickets Lost

Use cricket over notation in the app: `11.4` means 11 overs and 4 legal balls, not 11.4 decimal overs.

Output:

- Win Probability
- Lose Probability
- Visual probability bar
- Key engineered features used for the prediction

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

## Screenshots

After training the model and running the Flask app, add screenshots here:

- Home form with chase input.
- Prediction output with probability bar.
- EDA charts from `artifacts/`.

## Future Improvements

- Add player-level batter and bowler form.
- Add venue-specific par score and chasing history.
- Add toss, dew, season, and playoff pressure indicators.
- Add probability calibration with reliability curves.
- Use time-based validation by season.
- Track model drift as new IPL seasons are added.
- Add unit tests for feature calculations.

## Step-by-Step Implementation

1. Put `matches.csv` and `deliveries.csv` in `data/raw/`.
2. Run `python scripts/run_eda.py` to generate visual evidence.
3. Run `python scripts/train.py` to train and compare models.
4. Read `artifacts/metrics.csv` to justify model selection.
5. Read `artifacts/feature_importance.csv` and SHAP output to explain decisions.
6. Run `python app.py` to start the Flask app.
7. Discuss the project as a live ML system: data design, feature logic, leakage prevention, evaluation, explainability, and deployment.

For interview questions and detailed answers, see `docs/interview_prep.md`.
