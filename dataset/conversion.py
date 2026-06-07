import pandas as pd
import json

data = []

with open("yelp_academic_dataset_review.json") as f:
    for line in f:
        row = json.loads(line)
        data.append({
            "user_id": row["user_id"],
            "business_id": row["business_id"],
            "stars": row["stars"]
        })

df = pd.DataFrame(data)
df.to_csv("yelp.csv", index=False)
