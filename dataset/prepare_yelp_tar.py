import tarfile
import json
import pandas as pd
import os

# --------------------------------------------------
# PATH CONFIGURATION (FIXED)
# --------------------------------------------------
INPUT_TAR = "raw/yelp_dataset.tar"
EXTRACT_PATH = "raw/yelp_extracted"
OUTPUT_FILE = "processed/yelp.csv"

MAX_ROWS = 500000


def safe_extract_tar(tar, path="."):
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not os.path.abspath(member_path).startswith(os.path.abspath(path)):
            raise Exception("❌ Unsafe file detected in TAR archive!")
    tar.extractall(path=path, filter="data")


def extract_tar():
    print("Extracting Yelp dataset safely...")

    os.makedirs(EXTRACT_PATH, exist_ok=True)

    with tarfile.open(INPUT_TAR, "r") as tar:
        safe_extract_tar(tar, EXTRACT_PATH)

    print(f"✅ Extracted to: {EXTRACT_PATH}")


def find_review_file():
    print("Searching for review file...")

    for root, dirs, files in os.walk(EXTRACT_PATH):
        for file in files:
            if "review" in file and file.endswith(".json"):
                path = os.path.join(root, file)
                print(f"✅ Found review file: {path}")
                return path

    raise FileNotFoundError("❌ review.json file not found!")


def process_reviews(review_file):
    print("Reading review data...")

    rows = []

    with open(review_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):

            row = json.loads(line)

            rows.append({
                "user_id": row["user_id"],
                "item_id": row["business_id"],
                "rating": row["stars"]
            })

            if i >= MAX_ROWS:
                break

    df = pd.DataFrame(rows)

    print(f"✅ Loaded {len(df)} rows")
    return df


def clean_data(df):
    print("Cleaning dataset...")

    df = df.dropna()

    df = df.groupby("user_id").filter(lambda x: len(x) >= 5)
    df = df.groupby("item_id").filter(lambda x: len(x) >= 5)

    print(f"✅ After filtering: {len(df)} rows")
    return df


def save_data(df):
    os.makedirs("processed", exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ Saved dataset to: {OUTPUT_FILE}")


def prepare_yelp():

    print("====================================")
    print("   YELP DATASET PREPARATION")
    print("====================================")

    extract_tar()
    review_file = find_review_file()
    df = process_reviews(review_file)
    df = clean_data(df)
    save_data(df)

    print("\n✅ YELP DATASET READY!")


if __name__ == "__main__":
    prepare_yelp()
