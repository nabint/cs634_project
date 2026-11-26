"""See how the baseline models do when the training class mix changes."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from baseline_verification import RANDOM_STATE, load_datasets, make_preprocessor


RESULTS_CSV = Path("week2_results/class_imbalance_results.csv")
POSITIVE_SHARES = (5, 10, 20, 30, 40, 50)


def training_settings(dataset, negative_count, positive_count):
    positive_shares = (50, 40) if dataset == "Spambase" else POSITIVE_SHARES

    # The same training size has to work for every class mix.
    max_negative_share = 100 - min(positive_shares)
    max_positive_share = max(positive_shares)
    rows_from_negatives = negative_count * 100 // max_negative_share
    rows_from_positives = positive_count * 100 // max_positive_share
    train_size = min(rows_from_negatives, rows_from_positives)
    # Round down to a multiple of 20 so 5% is a whole number of rows.
    train_size -= train_size % 20

    settings = []
    for positive_share in positive_shares:
        positives = train_size * positive_share // 100
        negatives = train_size - positives
        setting = f"{100 - positive_share}/{positive_share}"
        settings.append((setting, negatives, positives))
    return settings


def evaluate_models(X_train, y_train, X_test, y_test):
    preprocess = make_preprocessor(X_train)
    train_features = preprocess.fit_transform(X_train)
    test_features = preprocess.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    for name, model in models.items():
        model.fit(train_features, y_train)
        predictions = model.predict(test_features)
        yield {
            "Algorithm": name,
            "Accuracy (%)": 100 * accuracy_score(y_test, predictions),
            "Precision (%)": 100 * precision_score(y_test, predictions, zero_division=0),
            "Recall (%)": 100 * recall_score(y_test, predictions, zero_division=0),
            "F1 (%)": 100 * f1_score(y_test, predictions, zero_division=0),
            "Missed Positives": int(((y_test == 1) & (predictions == 0)).sum()),
        }


def run_experiment(dataset, features, labels):
    # Set aside the test rows before sampling the training data.
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.20, stratify=labels, random_state=RANDOM_STATE
    )
    rng = np.random.default_rng(RANDOM_STATE)
    train_labels = y_train.to_numpy()
    negative_rows = rng.permutation(np.flatnonzero(train_labels == 0))
    positive_rows = rng.permutation(np.flatnonzero(train_labels == 1))

    # Make one 50/50 test set for all the runs on this dataset.
    test_labels = y_test.to_numpy()
    test_positive_rows = np.flatnonzero(test_labels == 1)
    test_negative_rows = np.flatnonzero(test_labels == 0)
    per_class = len(test_positive_rows)
    test_negatives = rng.choice(test_negative_rows, size=per_class, replace=False)
    test_rows = np.sort(np.concatenate([test_negatives, test_positive_rows]))
    X_test, y_test = X_test.iloc[test_rows], y_test.iloc[test_rows]

    results = []
    settings = training_settings(dataset, len(negative_rows), len(positive_rows))
    for setting, negatives, positives in settings:
        print(f"{dataset} | {setting} negative/positive", flush=True)
        selected_rows = np.concatenate([negative_rows[:negatives], positive_rows[:positives]])
        train_rows = np.sort(selected_rows)
        train_size = len(train_rows)
        setup = {
            "Dataset": dataset,
            "Training Setting": setting,
            "Train Negatives": negatives,
            "Train Positives": positives,
            "Train Rows": train_size,
            "Train Negative Share (%)": 100 * negatives / train_size,
            "Train Positive Share (%)": 100 * positives / train_size,
            "Test Rows": len(y_test),
            "Test Negatives": per_class,
            "Test Positives": per_class,
        }
        sample_features = X_train.iloc[train_rows]
        sample_labels = y_train.iloc[train_rows]
        for scores in evaluate_models(sample_features, sample_labels, X_test, y_test):
            results.append({**setup, **scores})
    return results


def main():
    results = []
    for dataset, features, labels in load_datasets():
        results.extend(run_experiment(dataset, features, labels))

    table = pd.DataFrame(results).round(2)
    RESULTS_CSV.parent.mkdir(exist_ok=True)
    table.to_csv(RESULTS_CSV, index=False)
    print(f"Saved {len(table)} comparisons to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
