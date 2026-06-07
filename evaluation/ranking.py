import numpy as np

def recall_at_k(recommended, relevant, k=10):
    rec_k = recommended[:k]
    return len(set(rec_k) & set(relevant)) / len(relevant)
