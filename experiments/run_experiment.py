import os
import datetime
import numpy as np
import torch
from scipy.stats import ttest_ind

from data.loaders import DataLoader

# Models
from models.base_mf import BaseMF
from models.x_ar_fmf import XARFMF
from models.ncf import NCF
from models.adversarial import AdversarialMF
from models.fair_rerank import FairRerank
from models.explain_baseline import ExplainBaseline

# Evaluation
from evaluation.accuracy import rmse
from evaluation.fairness import exposure_disparity
from explainability.metrics import explanation_coverage

# Config
from experiments.config import DATASETS, XAR_PARAMS, MF_PARAMS, SAMPLE_SIZE


# ✅ Number of runs (IMPORTANT)
NUM_RUNS = 10


# --------------------------------------------------
# RESULT WRITING
# --------------------------------------------------
def write_result(file, text):
    print(text)
    file.write(text + "\n")


# --------------------------------------------------
# SINGLE RUN
# --------------------------------------------------
def run_once(name, path, loader):

    data, df = getattr(loader, f"load_{name}")(path)

    train, test = loader.train_test_split(data)
    user_g, item_g = loader.build_groups(df)

    n_users = df["user_id"].nunique()
    n_items = df["item_id"].nunique()

    # ------------------ MODELS ------------------

    # MF
    mf = BaseMF(n_users, n_items,
                k=MF_PARAMS["k"],
                lr=MF_PARAMS["lr"],
                reg=MF_PARAMS["reg"])
    mf.train(train, epochs=MF_PARAMS["epochs"])

    # NCF
    ncf = NCF(n_users, n_items)
    ncf.eval()

    # ADV
    adv = AdversarialMF(n_users, n_items)
    adv.eval()

    # XAR
    xar = XARFMF(
        n_users,
        n_items,
        k=XAR_PARAMS["k"],
        alpha=XAR_PARAMS["alpha"],
        beta=XAR_PARAMS["beta"],
        lr=XAR_PARAMS["lr"]
    )
    xar.train(train, user_g, item_g, epochs=XAR_PARAMS["epochs"])

    item_centroids = xar.compute_item_centroids(item_g)

    # ------------------ EVAL ------------------

    eval_data = test[:SAMPLE_SIZE]

    results = {
        "MF": [],
        "NCF": [],
        "ADV": [],
        "XAR": []
    }

    truths = []
    explain_scores = []

    for u, i, r in eval_data:

        p_mf = mf.predict(u, i)
        results["MF"].append(p_mf)

        u_t = torch.tensor([u])
        i_t = torch.tensor([i])

        p_ncf = ncf(u_t, i_t).item()
        results["NCF"].append(p_ncf)

        p_adv, _ = adv(u_t, i_t)
        results["ADV"].append(p_adv.item())

        p_xar = xar.predict(u, i)
        results["XAR"].append(p_xar)

        truths.append(r)

        score = xar.explanation_score(i, item_g, item_centroids)
        explain_scores.append(score)

    # Metrics
    rmse_scores = {m: rmse(results[m], truths) for m in results}

    disparity = exposure_disparity(
        [i for _, i, _ in eval_data], item_g
    )

    coverage = explanation_coverage(explain_scores)

    return rmse_scores, disparity, coverage


# --------------------------------------------------
# MULTI-RUN DATASET
# --------------------------------------------------
def run_single_dataset(name, path, loader, result_file):

    write_result(result_file, "\n===============================")
    write_result(result_file, f" DATASET: {name.upper()}")
    write_result(result_file, "===============================")

    all_rmse = {
        "MF": [],
        "NCF": [],
        "ADV": [],
        "XAR": []
    }

    disparities = []
    coverages = []

    # ✅ MULTIPLE RUNS
    for run in range(NUM_RUNS):
        print(f"\n--- Run {run+1}/{NUM_RUNS} ---")

        rmse_scores, disparity, coverage = run_once(name, path, loader)

        for model in all_rmse:
            all_rmse[model].append(rmse_scores[model])

        disparities.append(disparity)
        coverages.append(coverage)

    # --------------------------------------------------
    # MEAN + STD
    # --------------------------------------------------
    write_result(result_file, "\n--- Aggregated Results ---")

    write_result(result_file, f"{'Model':<20}{'RMSE (mean±std)':<25}")
    write_result(result_file, "-" * 45)

    for model in all_rmse:
        mean = np.mean(all_rmse[model])
        std = np.std(all_rmse[model])
        write_result(result_file, f"{model:<20}{mean:.4f} ± {std:.4f}")

    # --------------------------------------------------
    # T-TEST (XAR vs MF)
    # --------------------------------------------------
    t_stat, p_value = ttest_ind(all_rmse["XAR"], all_rmse["MF"])

    write_result(result_file, "\n--- Statistical Test (XAR vs MF) ---")
    write_result(result_file, f"t-statistic: {t_stat:.4f}")
    write_result(result_file, f"p-value: {p_value:.6f}")

    # Interpretation
    if p_value < 0.05:
        write_result(result_file, "✅ Statistically significant difference")
    else:
        write_result(result_file, "⚠️ Not statistically significant")

    # --------------------------------------------------
    # FAIRNESS & EXPLAINABILITY
    # --------------------------------------------------
    write_result(result_file, "\n--- Other Metrics ---")

    write_result(result_file,
        f"Exposure Disparity (mean): {np.mean(disparities):.4f}")

    write_result(result_file,
        f"Explainability Coverage (XAR): {np.mean(coverages):.4f}")


# --------------------------------------------------
# RUN ALL
# --------------------------------------------------
def run_all():

    loader = DataLoader()

    os.makedirs("experiments/results", exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = f"experiments/results/results_{timestamp}.txt"

    with open(file_path, "w", encoding="utf-8") as f:

        write_result(f, "====================================")
        write_result(f, "   MULTI-RUN EXPERIMENT RESULTS")
        write_result(f, "====================================")

        for name, path in DATASETS.items():
            run_single_dataset(name, path, loader, f)

        write_result(f, "\n✅ ALL EXPERIMENTS COMPLETED")

    print(f"\n📁 Results saved to: {file_path}")


if __name__ == "__main__":
    run_all()
