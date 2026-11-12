# NLP Sentiment Classifier

A beginner-friendly **sentiment analysis** project built with Python and
scikit-learn. It reads short product/movie-style reviews and predicts
whether they are **positive** or **negative** — using a classic NLP
pipeline: **TF-IDF** features + **Logistic Regression**.

No downloads are needed at runtime: the project ships with a small,
hand-written dataset, so it works out of the box.

## Features

- Clean, commented, modular code (`src/` package) — easy to read and extend
- Text cleaning (lowercasing, URL/noise removal) shared by training and prediction
- TF-IDF vectorizer with unigrams + bigrams and English stop words removed
- Proper evaluation on a held-out test split: accuracy, precision/recall/F1
  per class, and a confusion matrix
- One-command prediction for any new sentence, with a confidence score
- Reproducible split (fixed random seed, stratified by label)

## Project structure

```
nlp-sentiment-classifier/
├── data/
│   ├── reviews.csv        # 195 labeled reviews (97 positive / 98 negative)
│   └── README.md          # About the dataset + how to use a bigger one
├── models/
│   └── model.joblib       # Trained model (created by src/train.py)
├── src/
│   ├── preprocess.py      # Data loading + text cleaning
│   ├── train.py           # Train, evaluate, and save the model
│   └── evaluate.py        # Re-score the saved model on the test split
├── predict.py             # Predict the sentiment of one sentence
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

Requires Python 3.10+.

```bash
git clone https://github.com/rashedrana/nlp-sentiment-classifier.git
cd nlp-sentiment-classifier

python3 -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

### 1. Train the model

```bash
python -m src.train
```

Output (actual run on the bundled dataset):

```
Loaded 195 reviews (97 positive, 98 negative)
Train size: 156  |  Test size: 39

=== Test set results ===
Accuracy: 0.974

Classification report:
              precision    recall  f1-score   support

    negative      1.000     0.950     0.974        20
    positive      0.950     1.000     0.974        19

    accuracy                          0.974        39

Confusion matrix (rows = actual, columns = predicted):
                 negative   positive
      negative         19          1
      positive          0         19

Model saved to models/model.joblib
```

### 2. Evaluate the saved model again

```bash
python -m src.evaluate
```

This reloads `models/model.joblib` and scores it on the same held-out
test split, printing the same metrics as above.

### 3. Predict a new sentence

```bash
python predict.py "I absolutely loved this film, the acting was brilliant!"
```

```
Text:       "I absolutely loved this film, the acting was brilliant and the story kept me hooked!"
Sentiment:  positive
Confidence: 54.0%
```

```bash
python predict.py "Terrible product, it broke after two days and the quality is awful."
```

```
Text:       "Terrible product, it broke after two days and the quality is awful."
Sentiment:  negative
Confidence: 70.4%
```

Mixed reviews are the interesting case — the model weighs the positive
and negative words against each other:

```bash
python predict.py "The camera is excellent and the screen is beautiful, but the battery life is disappointing and it feels a bit slow."
# -> positive, 54.7% confidence
```

## How it works

1. **Cleaning** (`src/preprocess.py`) — reviews are lowercased and stripped
   of URLs and noise so that "Great" and "great" are treated as the same word.
2. **Train/test split** — 80% of the reviews are used for training and 20%
   are held out for honest evaluation. The split is *stratified* (both
   parts keep the positive/negative balance) and uses a fixed seed, so
   every run gives the same result.
3. **TF-IDF** — each review becomes a vector of word scores. Words that
   appear in almost every review ("the", "this product") get low scores;
   distinctive words ("excellent", "terrible", "waste") get high scores.
   Two-word phrases ("waste of", "highly recommend") are included too.
4. **Logistic Regression** — learns a weight for every word/phrase:
   positive weights push towards *positive*, negative weights towards
   *negative*. A new review is scored by adding up the weights of its
   words, and `predict_proba` turns that score into the confidence %.
5. **Evaluation** — the held-out reviews are predicted and compared with
   their true labels: accuracy, per-class precision/recall/F1, and a
   confusion matrix showing exactly which reviews were misclassified.

## Next steps

- Train on a large real dataset (IMDb or Amazon reviews — see `data/README.md`)
- Try other models (Naive Bayes, Linear SVM) and compare them
- Add a third class, *neutral*, and handle negation ("not good") better
- Wrap the model in a small Flask/FastAPI web app
- Experiment with word embeddings or a transformer model (e.g. DistilBERT)

## License

MIT — see [LICENSE](LICENSE). © 2026 Rashed Rana
