import numpy as np

def explanation_coverage(scores, threshold=0.01):
    count = sum(1 for s in scores if s > threshold)
    return count / len(scores)

def explanation_fidelity(scores, fairness_signal):
    return np.corrcoef(scores, fairness_signal)[0,1]