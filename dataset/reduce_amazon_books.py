import pandas as pd
import os

# --------------------------------------------------
# PATH CONFIGURATION (inside data/ folder)
# --------------------------------------------------
INPUT_FILE = "raw/Books.csv"
OUTPUT_FILE = "processed/amazon.csv"
TARGET_SIZE = 200_000


def reduce_amazon_books():

    print("====================================")
    print("   AMAZON DATASET PREPARATION")
    print("====================================")

    # Ensure output folder exists
    os.makedirs("processed", exist_ok=True)

    # --------------------------------------------------
    # ✅ LOAD DATA (NO HEADER!)
    # --------------------------------------------------
    print("Loading dataset...")

    df = pd.read_csv(INPUT_FILE, header=None)

    print(f"Original dataset size: {len(df)} rows")

    # --------------------------------------------------
    # ✅ ASSIGN CORRECT COLUMN NAMES
    # Format: ISBN, UserID, Rating, Timestamp
    # --------------------------------------------------
    df.columns = ["item_id", "user_id", "rating", "timestamp"]

    print("Columns assigned:", df.columns.tolist())

    # --------------------------------------------------
    # ✅ SELECT RELEVANT COLUMNS
    # --------------------------------------------------
    df = df[["user_id", "item_id", "rating"]]

    # --------------------------------------------------
    # ✅ CLEAN DATA
    # --------------------------------------------------
    df = df.dropna()

    # Ensure rating is numeric
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna()

    # Remove invalid ratings
    df = df[df["rating"] > 0]

    print(f"After cleaning: {len(df)} rows")

    # --------------------------------------------------
    # ✅ FILTER SPARSE USERS / ITEMS (IMPORTANT)
    # NOTE: Use threshold = 3 (NOT 5!)
    # --------------------------------------------------
    print("Filtering sparse users/items...")

    df = df.groupby("user_id").filter(lambda x: len(x) >= 3)
    df = df.groupby("item_id").filter(lambda x: len(x) >= 3)

    print(f"After filtering: {len(df)} rows")

    print("Unique users:", df["user_id"].nunique())
    print("Unique items:", df["item_id"].nunique())

    # --------------------------------------------------
    # ✅ SAMPLE AFTER FILTERING (CORRECT ORDER)
    # --------------------------------------------------
    print("Sampling dataset...")

    if len(df) > TARGET_SIZE:
        df = df.sample(n=TARGET_SIZE, random_state=42)

    print(f"Final dataset size: {len(df)} rows")

    # --------------------------------------------------
    # ✅ SAVE FINAL DATASET
    # --------------------------------------------------
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Dataset saved to: {OUTPUT_FILE}")
    print("✅ AMAZON DATASET READY!")


# --------------------------------------------------
# RUN
# --------------------------------------------------
if __name__ == "__main__":
    reduce_amazon_books()