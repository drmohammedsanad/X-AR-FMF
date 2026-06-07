import os
import datetime
import numpy as np
import torch
from scipy.stats import ttest_ind

from data.loaders import DataLoader

# Models
from models.base_mf import BaseMF
from models.x_ar_fmf import XARFMF

# Evaluation
from evaluation.accuracy import rmse
from evaluation.fairness import exposure_disparity
from explainability.metrics import explanation_coverage

# Config
from experiments.config import DATASETS, MF_PARAMS, SAMPLE_SIZE


# ✅ Multi-run
NUM_RUNS = 10

# ✅ Hyperparameter grid
ALPHA_VALUES = [0.001, 0.01, 0.05, 0.1, 0.2]
BETA_VALUES = [0.001, 0.01, 0.05, 0.1, 0.2]


# --------------------------------------------------
def write_result(file, text):
    print(text)
    file.write(text + "\n")


# --------------------------------------------------
# SINGLE RUN
# --------------------------------------------------
def run_once(name, path, loader, alpha, beta):

    data, df = getattr(loader, f"load_{name}")(path)

    train, test = loader.train_test_split(data)
    user_g, item_g = loader.build_groups(df)

    n_users = df["user_id"].nunique()
    n_items = df["item_id"].nunique()

    # ------------------ MF ------------------
    mf = BaseMF(
        n_users,
        n_items,
        k=MF_PARAMS["k"],
        lr=MF_PARAMS["lr"],
        reg=MF_PARAMS["reg"]
    )
    mf.train(train, epochs=MF_PARAMS["epochs"])

    # ------------------ XAR (VARIABLE α, β) ------------------
    xar = XARFMF(
        n_users,
        n_items,
        k=MF_PARAMS["k"],
        alpha=alpha,
        beta=beta,
        lr=MF_PARAMS["lr"]
    )
    xar.train(train, user_g, item_g, epochs=MF_PARAMS["epochs"])

    centroids = xar.compute_item_centroids(item_g)

    # ------------------ EVAL ------------------
    eval_data = test[:SAMPLE_SIZE]

    preds_mf, preds_xar = [], []
    truths = []
    explain_scores = []

    for u, i, r in eval_data:

        preds_mf.append(mf.predict(u, i))
        preds_xar.append(xar.predict(u, i))

        truths.append(r)

        explain_scores.append(
            xar.explanation_score(i, item_g, centroids)
        )

    rmse_mf = rmse(preds_mf, truths)
    rmse_xar = rmse(preds_xar, truths)

    disparity = exposure_disparity(
        [i for _, i, _ in eval_data], item_g
    )

    coverage = explanation_coverage(explain_scores)

    return rmse_mf, rmse_xar, disparity, coverage


# --------------------------------------------------
# GRID SEARCH PER DATASET
# --------------------------------------------------
def run_single_dataset(name, path, loader, result_file):

    write_result(result_file, "\n===============================")
    write_result(result_file, f" DATASET: {name.upper()}")
    write_result(result_file, "===============================")

    for alpha in ALPHA_VALUES:
        for beta in BETA_VALUES:

            write_result(result_file,
                         f"\n--- Alpha={alpha}, Beta={beta} ---")

            mf_scores = []
            xar_scores = []
            disparities = []
            coverages = []

            for run in range(NUM_RUNS):
                print(f"Run {run+1}/{NUM_RUNS} (α={alpha}, β={beta})")

                res = run_once(name, path, loader, alpha, beta)

                rmse_mf, rmse_xar, disparity, coverage = res

                mf_scores.append(rmse_mf)
                xar_scores.append(rmse_xar)
                disparities.append(disparity)
                coverages.append(coverage)

            # ✅ Aggregate
            mf_mean, mf_std = np.mean(mf_scores), np.std(mf_scores)
            xar_mean, xar_std = np.mean(xar_scores), np.std(xar_scores)

            disp_mean = np.mean(disparities)
            cov_mean = np.mean(coverages)

            # ✅ T-test
            t_stat, p_val = ttest_ind(xar_scores, mf_scores)

            # ✅ OUTPUT
            write_result(result_file,
                f"MF  RMSE: {mf_mean:.4f} ± {mf_std:.4f}")
            write_result(result_file,
                f"XAR RMSE: {xar_mean:.4f} ± {xar_std:.4f}")

            write_result(result_file,
                f"p-value: {p_val:.6f}")

            write_result(result_file,
                f"Exposure: {disp_mean:.2f} | Coverage: {cov_mean:.4f}")

            # Interpretation flag
            if p_val < 0.05:
                write_result(result_file, "✅ Significant difference")
            else:
                write_result(result_file, "⚠️ Not significant")


# --------------------------------------------------
# RUN ALL
# --------------------------------------------------
def run_all():

    loader = DataLoader()

    os.makedirs("experiments/results", exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = f"experiments/results/alpha_beta_results_{timestamp}.txt"

    with open(file_path, "w", encoding="utf-8") as f:

        write_result(f, "====================================")
        write_result(f, "   ALPHA-BETA TUNING RESULTS")
        write_result(f, "====================================")

        for name, path in DATASETS.items():
            run_single_dataset(name, path, loader, f)

        write_result(f, "\n✅ TUNING COMPLETED")

    print(f"\n📁 Results saved to: {file_path}")


if __name__ == "__main__":
    run_all()