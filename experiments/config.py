"""
Central configuration file for experiments
"""

# --------------------------------------------------
# DATASETS
# --------------------------------------------------
DATASETS = {
    "amazon": "data/processed/amazon.csv",
    "yelp": "data/processed/yelp.csv",
    "douban": "data/processed/douban.csv"
}

# --------------------------------------------------
# DATASET SETTINGS
# --------------------------------------------------
TEST_RATIO = 0.2
SAMPLE_SIZE = 2000   # used in evaluation loops

# --------------------------------------------------
# MODEL PARAMETERS
# --------------------------------------------------

# MF
MF_PARAMS = {
    "k": 20,
    "lr": 0.01,
    "reg": 0.01,
    "epochs": 10
}

# X-AR-FMF (your model)
XAR_PARAMS = {
    "k": 20,
    "alpha": 0.05,
    "beta": 0.05,
    "lr": 0.01,
    "epochs": 10
}

# --------------------------------------------------
# FAIRNESS SETTINGS
# --------------------------------------------------
MIN_INTERACTIONS = 3   # filtering threshold

# --------------------------------------------------
# EXPLAINABILITY SETTINGS
# --------------------------------------------------
EXPLANATION_THRESHOLD = 0.01

# --------------------------------------------------
# OUTPUT SETTINGS
# --------------------------------------------------
RESULTS_DIR = "experiments/results/"