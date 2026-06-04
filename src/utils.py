from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_object(obj: Any, path: Path) -> None:
    ensure_parent_dir(path)
    joblib.dump(obj, path)


def load_object(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return joblib.load(path)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    ensure_parent_dir(path)
    df.to_csv(path, index=False)
