
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

DATA_FILE = DATA_DIR / "SMSSpamCollection"
DATA_URL = (
    "https://archive.ics.uci.edu/static/public/228/"
    "sms+spam+collection.zip"
)


def load_data():
    if not DATA_FILE.exists():
        print("Downloading dataset...")
        with urlopen(DATA_URL, timeout=60) as response:
            contents = response.read()

        with ZipFile(BytesIO(contents)) as archive:
            filename = next(
                n for n in archive.namelist()
                if n.endswith("SMSSpamCollection")
            )
            DATA_FILE.write_bytes(archive.read(filename))

    df = pd.read_csv(
        DATA_FILE,
        sep="\t",
        header=None,
        names=["label", "text"],
        encoding="utf-8",
    )

    df = df.dropna()
    df["label"] = df["label"].str.strip().str.lower()
    df["text"] = df["text"].astype(str).str.strip()
    return df[df["label"].isin(["ham", "spam"]) & (df["text"] != "")]


def main():
    df = load_data()

    print(f"\nTotal messages: {len(df)}")
    print(df["label"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    classifiers = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "Linear SVM": LinearSVC(
            class_weight="balanced",
            random_state=42,
        ),
    }

    results = []
    best_model = None
    best_name = ""
    best_f1 = -1

    for name, classifier in classifiers.items():
        print(f"\nTraining {name}...")

        model = Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                max_features=30000,
                sublinear_tf=True,
            )),
            ("classifier", classifier),
        ])

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, pos_label="spam", zero_division=0
        )
        recall = recall_score(
            y_test, predictions, pos_label="spam", zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, pos_label="spam", zero_division=0
        )

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Spam precision: {precision:.4f}")
        print(f"Spam recall: {recall:.4f}")
        print(f"Spam F1-score: {f1:.4f}")

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Spam Precision": precision,
            "Spam Recall": recall,
            "Spam F1": f1,
        })

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name

    results_df = pd.DataFrame(results).sort_values(
        "Spam F1", ascending=False
    )
    results_df.to_csv(DATA_DIR / "model_results.csv", index=False)

    print("\n=== MODEL COMPARISON ===")
    print(results_df.to_string(index=False, float_format="%.4f"))

    predictions = best_model.predict(X_test)
    print(f"\nBest model by spam F1-score: {best_name}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions))
    print("Confusion matrix (labels: ham, spam):")
    print(confusion_matrix(y_test, predictions, labels=["ham", "spam"]))

    joblib.dump({
        "pipeline": best_model,
        "model_name": best_name,
        "dataset": "UCI SMS Spam Collection",
    }, MODEL_DIR / "spam_model.joblib")

    print("\nModel saved in models/spam_model.joblib")
    print("Comparison saved in data/model_results.csv")

    print("\n=== SAMPLE PREDICTIONS ===")
    samples = [
        "Congratulations! You have won a free prize. Click here now!",
        "Hi, are we still meeting for lunch tomorrow?",
    ]

    for message in samples:
        prediction = best_model.predict([message])[0]
        label = "SPAM" if prediction == "spam" else "NOT SPAM"
        print(f"\nMessage: {message}")
        print(f"Prediction: {label}")


if __name__ == "__main__":
    main()
