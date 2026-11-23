from pathlib import Path

import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_DIR = Path("data")
OUTPUT_DIR = Path("week1_results")
RANDOM_STATE = 42


def load_datasets():
    credit = pd.read_csv(DATA_DIR / "creditcard.csv")
    yield "Credit Card Fraud", credit.drop(columns="Class"), credit["Class"].astype(int)

    bank = pd.read_csv(DATA_DIR / "bank-full.csv")
    subscribed = (bank["y"] == "yes").astype(int)
    # Call duration is only known after the call ends.
    yield "Bank Marketing", bank.drop(columns=["y", "duration"]), subscribed

    spam = pd.read_csv(DATA_DIR / "spambase.csv")
    yield "Spambase", spam.drop(columns="spam"), spam["spam"].astype(int)


def make_preprocessor(X: pd.DataFrame):
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    text_columns = X.select_dtypes(exclude="number").columns.tolist()
    numeric_steps = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    text_steps = make_pipeline(
        SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")
    )
    return make_column_transformer(
        (numeric_steps, numeric_columns),
        (text_steps, text_columns),
    )


def train_and_evaluate(name: str, X: pd.DataFrame, y: pd.Series) -> list[dict]:
    train_features, test_features, train_labels, test_labels = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    results = []
    for model_name, model in models.items():
        pipeline = make_pipeline(make_preprocessor(train_features), model)
        pipeline.fit(train_features, train_labels)
        predictions = pipeline.predict(test_features)

        accuracy = accuracy_score(test_labels, predictions) * 100
        precision = precision_score(test_labels, predictions, zero_division=0) * 100
        recall = recall_score(test_labels, predictions, zero_division=0) * 100
        f1 = f1_score(test_labels, predictions, zero_division=0) * 100

        results.append({
            "Dataset": name,
            "Algorithm": model_name,
            "Accuracy (%)": round(accuracy, 2),
            "Precision (%)": round(precision, 2),
            "Recall (%)": round(recall, 2),
            "F1 (%)": round(f1, 2),
        })

    return results


def main():
    results = []
    for name, features, labels in load_datasets():
        print(f"Training three models on {name}...", flush=True)
        results.extend(train_and_evaluate(name, features, labels))

    results_df = pd.DataFrame(results)
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "initial_model_baseline_results.csv"
    results_df.to_csv(output_path, index=False)

    print("\nInitial Baseline Results (80/20 stratified train/test split):")
    print(results_df.to_string(index=False))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
