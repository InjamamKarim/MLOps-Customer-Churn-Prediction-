import pandas as pd
from pathlib import Path
import requests
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RAW.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
CSV_PATH = RAW / "Telco-Customer-Churn.csv"

def download():
    if not CSV_PATH.exists():
        print("Downloading dataset...")
        r = requests.get(URL)
        r.raise_for_status()
        CSV_PATH.write_bytes(r.content)
    return CSV_PATH

def prepare():
    df = pd.read_csv(download())
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    train, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["Churn"]
    )
    train.to_csv(PROCESSED / "train.csv", index=False)
    test.to_csv(PROCESSED / "test.csv", index=False)
    print(f"Saved train {train.shape} and test {test.shape}")

if __name__ == "__main__":
    prepare()