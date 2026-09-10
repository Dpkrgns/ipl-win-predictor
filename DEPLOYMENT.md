# Deployment and API

## Local production-style run

After installing dependencies and training the model:

```powershell
python app.py
```

For a Windows production process, install and run Waitress:

```powershell
python -m pip install waitress
waitress-serve --listen=127.0.0.1:5000 app:app
```

For Linux hosting platforms:

```bash
python -m pip install gunicorn
gunicorn --workers 2 --threads 4 --timeout 120 app:app
```

The application needs the trained artifact at `artifacts/ipl_win_pipeline.joblib`. Generate it with:

```bash
python scripts/train.py
```

## API

### Health

```http
GET /api/health
```

Returns model readiness:

```json
{"status": "ok", "model_ready": true}
```

### Prediction

```http
POST /api/predict
Content-Type: application/json
```

Example request:

```json
{
  "batting_team": "Chennai Super Kings",
  "bowling_team": "Delhi Capitals",
  "venue": "Wankhede Stadium",
  "target_score": 178,
  "score": 92,
  "overs_completed": "11.4",
  "wickets_lost": 3
}
```

The response includes the headline probability, engineered match features, SHAP explanation signals, projected probability timeline, and counterfactual scenarios.

Example PowerShell request:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:5000/api/predict `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"batting_team":"Chennai Super Kings","bowling_team":"Delhi Capitals","venue":"Wankhede Stadium","target_score":178,"score":92,"overs_completed":"11.4","wickets_lost":3}'
```

### Evaluation

```http
GET /api/evaluation
```

Returns `metrics.csv` and `feature_importance.csv` when training artifacts exist.

## Hosting checklist

1. Set the Python version to 3.11 or 3.12.
2. Install `requirements.txt`.
3. Provide `data/raw/matches.csv` and `data/raw/deliveries.csv` during the build step, or provide a pre-trained artifact.
4. Run `python scripts/train.py` once during setup.
5. Start the service with `gunicorn app:app` on Linux or Waitress on Windows.
6. Keep raw datasets private when redistribution is restricted.
