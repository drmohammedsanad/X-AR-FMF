import pandas as pd
import json


# ==================================================
# COMMON CLEANING FUNCTION
# ==================================================
def clean_dataframe(df, user_col, item_col, rating_col, min_interactions=5):

    # Keep only required columns
    df = df[[user_col, item_col, rating_col]].copy()

    # Rename columns to unified format
    df.columns = ["user_id", "item_id", "rating"]

    # ✅ Remove missing values
    df = df.dropna()

    # ✅ Remove sparse users
    df = df.groupby("user_id").filter(lambda x: len(x) >= min_interactions)

    # ✅ Remove sparse items
    df = df.groupby("item_id").filter(lambda x: len(x) >= min_interactions)

    return df


# ==================================================
# AMAZON DATASET
# ==================================================
def process_amazon():

    print("\nProcessing Amazon dataset...")

    # Raw file (after download + reduction)
    df = pd.read_csv("data/amazon.csv")

    df = clean_dataframe(
        df,
        user_col="reviewerID",
        item_col="asin",
        rating_col="overall"
    )

    # Save clean version
    df.to_csv("data/amazon.csv", index=False)

    print(f"✅ Amazon ready: {len(df)} rows")


# ==================================================
# YELP DATASET
# ==================================================
def process_yelp():

    print("\nProcessing Yelp dataset...")

    input_file = "data/yelp_raw.json"
    output_file = "data/yelp.csv"

    rows = []

    print("Reading JSON... (this may take time)")

    with open(input_file, "r") as f:
        for i, line in enumerate(f):
            row = json.loads(line)

            rows.append({
                "user_id": row["user_id"],
                "item_id": row["business_id"],
                "rating": row["stars"]
            })

            # limit to manageable size (optional)
            if i > 300000:
                break

    df = pd.DataFrame(rows)

    df = clean_dataframe(
        df,
        user_col="user_id",
        item_col="item_id",
        rating_col="rating"
    )

    df.to_csv(output_file, index=False)

    print(f"✅ Yelp ready: {len(df)} rows")


# ==================================================
# DOUBAN DATASET
# ==================================================
def process_douban():

    print("\nProcessing Douban dataset...")

    df = pd.read_csv("data/douban_raw.csv")

    df = clean_dataframe(
        df,
        user_col="user",
        item_col="movie",
        rating_col="rating"
    )

    df.to_csv("data/douban.csv", index=False)

    print(f"✅ Douban ready: {len(df)} rows")


# ==================================================
# MAIN PIPELINE
# ==================================================
if __name__ == "__main__":

    print("====================================")
    print("   DATASET PREPARATION PIPELINE")
    print("====================================")

    process_amazon()
    process_yelp()
    process_douban()

    print("\n✅ ALL DATASETS ARE READY!")