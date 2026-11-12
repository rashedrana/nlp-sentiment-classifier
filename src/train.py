"""Train the sentiment classifier and save it to disk.

Usage:
    python -m src.train

What it does:
    1. Loads and cleans the reviews from data/reviews.csv.
    2. Splits them into a training set (80%) and a held-out test set (20%).
    3. Trains a TF-IDF + Logistic Regression pipeline on the training set.
    4. Prints evaluation metrics on the test set (never seen during training).
    5. Saves the trained pipeline to models/model.joblib so predict.py and
       src/evaluate.py can load it.
"""

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import DEFAULT_DATA_PATH, load_data

# Shared split settings. src/evaluate.py uses the same values so it scores
# the model on exactly the same held-out test set.
TEST_SIZE = 0.2
RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"


def build_pipeline() -> Pipeline:
    """Create the (untrained) ML pipeline.

    TF-IDF turns each review into a vector of word scores: common words
    count for little, distinctive words count for a lot. Unigrams + bigrams
    (single words and two-word phrases like "not good") are used.

    Logistic Regression then learns which words push the prediction
    towards "positive" and which towards "negative".
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",   # ignore "the", "a", "is", ...
                    ngram_range=(1, 2),     # words + two-word phrases
                    min_df=1,
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1000),
            ),
        ]
    )


def split_data(df):
    """Split the DataFrame into train/test parts (stratified by label)."""
    return train_test_split(
        df["review"],
        df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )


def print_metrics(y_true, y_pred, title: str) -> None:
    """Print accuracy, per-class precision/recall/F1, and a confusion matrix."""
    print(f"\n=== {title} ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=3))
    labels = ["negative", "positive"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(f"               {labels[0]:>10} {labels[1]:>10}")
    for label, row in zip(labels, cm):
        print(f"  {label:>12} {row[0]:>10} {row[1]:>10}")


def main() -> None:
    df = load_data(DEFAULT_DATA_PATH)
    print(f"Loaded {len(df)} reviews "
          f"({(df['label'] == 'positive').sum()} positive, "
          f"{(df['label'] == 'negative').sum()} negative)")

    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    model = build_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print_metrics(y_test, y_pred, "Test set results")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
