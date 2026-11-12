"""Predict the sentiment of a single sentence.

Usage:
    python predict.py "I really loved this movie, it was wonderful!"

Prints the predicted sentiment (positive/negative) and the model's
confidence. Run src/train.py first so models/model.joblib exists.
"""

import sys
from pathlib import Path

import joblib

# Allow running this file directly from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.preprocess import clean_text  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parent / "models" / "model.joblib"


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python predict.py "your review text here"')
        raise SystemExit(1)

    if not MODEL_PATH.exists():
        print(f"No saved model found at {MODEL_PATH}.")
        print("Train it first with:  python -m src.train")
        raise SystemExit(1)

    text = " ".join(sys.argv[1:])
    model = joblib.load(MODEL_PATH)

    cleaned = clean_text(text)
    prediction = model.predict([cleaned])[0]

    # predict_proba gives one probability per class; pick the one that
    # belongs to the predicted label and show it as the confidence.
    probabilities = model.predict_proba([cleaned])[0]
    confidence = dict(zip(model.classes_, probabilities))[prediction]

    print(f'Text:       "{text}"')
    print(f"Sentiment:  {prediction}")
    print(f"Confidence: {confidence:.1%}")


if __name__ == "__main__":
    main()
