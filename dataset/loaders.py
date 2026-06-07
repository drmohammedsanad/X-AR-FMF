import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DataLoader:

    def __init__(self):
        pass

    # ------------------------------------------
    # GENERIC PROCESSOR
    # ------------------------------------------
    def _process(self, df, user_col, item_col, rating_col):

        df = df[[user_col, item_col, rating_col]].copy()

        # ✅ Use fresh encoders per dataset
        user_enc = LabelEncoder()
        item_enc = LabelEncoder()

        df[user_col] = user_enc.fit_transform(df[user_col])
        df[item_col] = item_enc.fit_transform(df[item_col])

        df.columns = ["user_id", "item_id", "rating"]

        data = list(zip(df['user_id'], df['item_id'], df['rating']))

        return data, df

    # ------------------------------------------
    # DATASET LOADERS (FIXED)
    # ------------------------------------------
    def load_amazon(self, path):
        df = pd.read_csv(path)
        return self._process(df, "user_id", "item_id", "rating")

    def load_yelp(self, path):
        df = pd.read_csv(path)
        return self._process(df, "user_id", "item_id", "rating")

    def load_douban(self, path):
        df = pd.read_csv(path)
        return self._process(df, "user_id", "item_id", "rating")

    # ------------------------------------------
    # TRAIN TEST SPLIT
    # ------------------------------------------
    def train_test_split(self, data, test_ratio=0.2):
        np.random.shuffle(data)
        split = int(len(data)*(1-test_ratio))
        return data[:split], data[split:]

    # ------------------------------------------
    # GROUPS (FAIRNESS)
    # ------------------------------------------
    def build_groups(self, df):

        user_counts = df.groupby('user_id').size()
        item_counts = df.groupby('item_id').size()

        user_thr = user_counts.median()
        item_thr = item_counts.median()

        user_groups = {u: int(user_counts[u] > user_thr) for u in user_counts.index}
        item_groups = {i: int(item_counts[i] > item_thr) for i in item_counts.index}

        return user_groups, item_groups