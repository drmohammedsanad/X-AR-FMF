import pandas as pd
import os

# --------------------------------------------------
# PATHS
# --------------------------------------------------
INPUT_FILE = "raw/douban_raw.txt"
OUTPUT_FILE = "processed/douban.csv"

TARGET_SIZE = 200_000


def prepare_douban():

    print("====================================")
    print("   DOUBAN DATASET PREPARATION")
    print("====================================")

    os.makedirs("processed", exist_ok=True)

    # --------------------------------------------------
    # ✅ LOAD DATA (CORRECT FORMAT)
    # --------------------------------------------------
    print("Loading dataset...")

    df = pd.read_csv(
        INPUT_FILE,
        sep="\t",       # ✅ TAB-separated
        header=0,       # ✅ HAS HEADER
        engine="python"
    )

    print(f"Original rows: {len(df)}")
    print("Columns:", df.columns.tolist())

    # --------------------------------------------------
    # ✅ SELECT CORRECT COLUMNS
    # --------------------------------------------------
    df = df[["user_id", "movie_id", "rating"]]

    # Rename to match pipeline
    df.columns = ["user_id", "item_id", "rating"]

    # --------------------------------------------------
    # ✅ CLEAN DATA
    # --------------------------------------------------
    df = df.dropna()

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna()

    df = df[df["rating"] > 0]

    print(f"After cleaning: {len(df)} rows")

    # --------------------------------------------------
    # ✅ FILTER SPARSE DATA
    # --------------------------------------------------
    print("Filtering sparse users/items...")

    df = df.groupby("user_id").filter(lambda x: len(x) >= 3)
    df = df.groupby("item_id").filter(lambda x: len(x) >= 3)

    print(f"After filtering: {len(df)} rows")

    print("Unique users:", df["user_id"].nunique())
    print("Unique items:", df["item_id"].nunique())

    # --------------------------------------------------
    # ✅ SAMPLE
    # --------------------------------------------------
    if len(df) > TARGET_SIZE:
        df = df.sample(n=TARGET_SIZE, random_state=42)

    print(f"Final dataset size: {len(df)}")

    # --------------------------------------------------
    # ✅ SAVE
    # --------------------------------------------------
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Saved to: {OUTPUT_FILE}")
    print("✅ DOUBAN DATASET READY!")


# --------------------------------------------------
# RUN
# --------------------------------------------------
if __name__ == "__main__":
    prepare_douban()