import numpy as np

def rmse(preds, truths):
    return np.sqrt(np.mean((np.array(preds)-np.array(truths))**2))