"""Evaluate the saved model on the held-out test split.

Usage:
    python -m src.evaluate

Loads models/model.joblib (created by src/train.py) and scores it on the
same test split the training script used, so the numbers are comparable.
"""

import joblib

from src.preprocess import DEFAULT_DATA_PATH, load_data
from src.train import MODEL_PATH, print_metrics, split_data


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"No saved model found at {MODEL_PATH}. "
            "Run 'python -m src.train' first."
        )

    model = joblib.load(MODEL_PATH)
    df = load_data(DEFAULT_DATA_PATH)
    _, X_test, _, y_test = split_data(df)

    print(f"Evaluating saved model on {len(X_test)} held-out reviews...")
    y_pred = model.predict(X_test)
    print_metrics(y_test, y_pred, "Saved model — test set results")


if __name__ == "__main__":
    main()
