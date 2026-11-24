"""Download the three datasets and save them as CSV files in data/."""

import csv
from io import BytesIO, TextIOWrapper
from pathlib import Path
from subprocess import run
from tempfile import TemporaryDirectory
from zipfile import ZipFile


DATA_DIR = Path("data")
CREDIT_URL = "https://www.kaggle.com/api/v1/datasets/download/mlg-ulb/creditcardfraud"
BANK_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
SPAM_URL = "https://archive.ics.uci.edu/static/public/94/spambase.zip"


def download(url, path):
    run(
        ["curl", "--fail", "--location", "--silent", "--show-error", "--output", str(path), url],
        check=True,
    )


def get_credit_card(folder):
    print("Downloading Credit Card Fraud...", flush=True)
    archive = folder / "creditcard.zip"
    download(CREDIT_URL, archive)
    with ZipFile(archive) as dataset:
        dataset.extract("creditcard.csv", folder)


def get_bank_marketing(folder):
    print("Downloading Bank Marketing...", flush=True)
    archive = folder / "bank_marketing.zip"
    download(BANK_URL, archive)
    with ZipFile(archive) as dataset:
        bank_zip = BytesIO(dataset.read("bank.zip"))

    # The original Bank Marketing file uses semicolons.
    with ZipFile(bank_zip) as bank, bank.open("bank-full.csv") as source:
        rows = csv.reader(TextIOWrapper(source, encoding="utf-8"), delimiter=";")
        with (folder / "bank-full.csv").open("w", encoding="utf-8", newline="") as output:
            csv.writer(output, lineterminator="\n").writerows(rows)


def get_spambase(folder):
    print("Downloading Spambase...", flush=True)
    archive = folder / "spambase.zip"
    download(SPAM_URL, archive)
    with ZipFile(archive) as dataset:
        names = dataset.read("spambase.names").decode("ascii").splitlines()
        rows = dataset.read("spambase.data").decode("ascii").splitlines()

    # The feature names are stored separately from the data.
    columns = [line.split(":", 1)[0].strip() for line in names if line.endswith("continuous.")]
    with (folder / "spambase.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(columns + ["spam"])
        writer.writerows(csv.reader(rows))


def main():
    DATA_DIR.mkdir(exist_ok=True)
    # Finish all three downloads before replacing any existing CSVs.
    with TemporaryDirectory(dir=DATA_DIR) as temp_dir:
        folder = Path(temp_dir)
        get_credit_card(folder)
        get_bank_marketing(folder)
        get_spambase(folder)
        for filename in ("creditcard.csv", "bank-full.csv", "spambase.csv"):
            path = DATA_DIR / filename
            (folder / filename).replace(path)
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
