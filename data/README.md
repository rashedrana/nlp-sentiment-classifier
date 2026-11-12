# About the dataset

`reviews.csv` contains **195 short reviews** with a label for each one:

| Column   | Meaning                                             |
|----------|-----------------------------------------------------|
| `review` | The review text (1–2 sentences, product/movie style) |
| `label`  | `positive` or `negative`                            |

The set is (nearly) balanced: **97 positive** and **98 negative** reviews, written by
hand for this project so the repo works out of the box with no downloads.

## Using a bigger dataset

The small built-in set is great for learning, but a real project would use
thousands of reviews. To swap it:

1. Prepare a CSV with the same two columns (`review`, `label`) and the same
   label values (`positive` / `negative`).
2. Save it as `data/reviews.csv` (or point `load_data()` in
   `src/preprocess.py` at your file).
3. Re-run `python -m src.train`.

Good public options include the IMDb movie reviews dataset and the Amazon
product reviews datasets — just map their labels to `positive`/`negative`
first (for example, ratings of 4–5 stars → `positive`, 1–2 stars →
`negative`).
