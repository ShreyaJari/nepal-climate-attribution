"""
multiseed_robustness.py

Checks whether the hyperparameter and anomaly-score selection from
compare_hyperparameters.py is stable across different train/validation
splits, or whether it happened to depend on the one random seed used
originally. Same grid, same discriminability-based selection criterion,
repeated across several seeds -- each seed gets its own train/validation
split, its own normalization stats, and its own full sweep.

This does NOT overwrite the production model files in models/. Its job is
to report on robustness, not to re-select the model score_anomaly.py uses.
If this run changes which configuration should be considered "best," that
is a decision to make deliberately (re-running train_vae.py and
score_anomaly.py with the new choice), not something this script does
automatically.

Requirements
------------
    pip install torch numpy pandas matplotlib
Must be run from the same scripts/ folder as train_vae.py and
compare_hyperparameters.py (imports from both).
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from train_vae import ConvVAE, FieldsDataset, load_all_years, split_years, stack_years, DEVICE, BATCH_SIZE
from compare_hyperparameters import (
    train_one_config, compute_per_sample_scores, effect_size,
    BETA_VALUES, LATENT_DIMS, NUM_EPOCHS_SWEEP, KL_COLLAPSE_THRESHOLD,
    SYNTHETIC_SHIFT_MAGNITUDES_DEGC, SELECTION_SHIFT_MAGNITUDE, N_VALIDATION_YEARS,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

SEEDS = [42, 1, 7, 123, 2024]

# The configuration currently in production (models/vae_best.pt), so its
# consistency across seeds can be checked specifically, not just whichever
# configuration happens to win each time.
PRODUCTION_BETA = 0.1
PRODUCTION_LATENT_DIM = 8
PRODUCTION_SCORE_TYPE = "kl"


def print_section(title):
    print("\n" + "=" * 78)
    print(title)


def run_one_seed(data_by_year, seed):
    training_years, validation_years = split_years(data_by_year, N_VALIDATION_YEARS, seed)
    train_fields_raw = stack_years(data_by_year, training_years)
    val_fields_raw = stack_years(data_by_year, validation_years)

    pixel_mean = np.nanmean(train_fields_raw, axis=0)
    pixel_std = np.nanstd(train_fields_raw, axis=0)
    pixel_std = np.where(pixel_std < 1e-6, 1e-6, pixel_std)
    train_fields_norm = np.nan_to_num((train_fields_raw - pixel_mean) / pixel_std, nan=0.0)
    val_fields_norm = np.nan_to_num((val_fields_raw - pixel_mean) / pixel_std, nan=0.0)

    train_loader = DataLoader(FieldsDataset(train_fields_norm), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(FieldsDataset(val_fields_norm), batch_size=BATCH_SIZE, shuffle=False)
    input_size = train_fields_raw.shape[-1]

    seed_results = []
    for beta in BETA_VALUES:
        for latent_dim in LATENT_DIMS:
            model, history = train_one_config(latent_dim, beta, NUM_EPOCHS_SWEEP, train_loader, val_loader, input_size)
            final_val_kl = np.mean(history["val_kl"][-5:])
            collapsed = final_val_kl < KL_COLLAPSE_THRESHOLD

            normal_recon, normal_kl, normal_combined = compute_per_sample_scores(model, val_fields_norm)
            row = {"seed": seed, "beta": beta, "latent_dim": latent_dim, "posterior_collapse_flag": collapsed}

            for shift in SYNTHETIC_SHIFT_MAGNITUDES_DEGC:
                shifted_raw = val_fields_raw + shift
                shifted_norm = np.nan_to_num((shifted_raw - pixel_mean) / pixel_std, nan=0.0)
                anom_recon, anom_kl, anom_combined = compute_per_sample_scores(model, shifted_norm)
                row[f"d_recon_{shift}degc"] = effect_size(anom_recon, normal_recon)
                row[f"d_kl_{shift}degc"] = effect_size(anom_kl, normal_kl)
                row[f"d_combined_{shift}degc"] = effect_size(anom_combined, normal_combined)

            seed_results.append(row)

    return pd.DataFrame(seed_results)


def main():
    data_by_year = load_all_years()

    print_section(f"Running the full sweep across {len(SEEDS)} seeds: {SEEDS}")
    print(f"Each seed gets its own train/validation split, own normalization, own {len(BETA_VALUES) * len(LATENT_DIMS)}-configuration sweep.")

    all_results = []
    start_time = time.time()
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        seed_df = run_one_seed(data_by_year, seed)
        all_results.append(seed_df)
    elapsed_minutes = (time.time() - start_time) / 60
    print(f"\nAll seeds complete. Elapsed: {elapsed_minutes:.1f} minutes.")

    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(RESULTS_DIR / "vae_multiseed_robustness.csv", index=False)
    print(f"Saved full results to {RESULTS_DIR / 'vae_multiseed_robustness.csv'}")

    print_section("Winning configuration per seed")
    s = SELECTION_SHIFT_MAGNITUDE
    score_cols = [f"d_recon_{s}degc", f"d_kl_{s}degc", f"d_combined_{s}degc"]

    winners = []
    for seed in SEEDS:
        seed_df = combined_df[combined_df["seed"] == seed]
        non_collapsed = seed_df[~seed_df["posterior_collapse_flag"]]
        if non_collapsed.empty:
            print(f"  seed {seed}: every configuration showed possible posterior collapse, skipping")
            continue
        best_per_score = {col: non_collapsed.loc[non_collapsed[col].idxmax()] for col in score_cols}
        best_col = max(score_cols, key=lambda c: best_per_score[c][c])
        best_row = best_per_score[best_col]
        score_type = best_col.replace(f"_{s}degc", "").replace("d_", "")
        print(f"  seed {seed}: winner = {score_type}, beta={best_row['beta']}, "
              f"latent_dim={int(best_row['latent_dim'])}, d={best_row[best_col]:.2f}")
        winners.append({"seed": seed, "score_type": score_type, "beta": best_row["beta"],
                         "latent_dim": int(best_row["latent_dim"]), "effect_size": best_row[best_col]})

    winners_df = pd.DataFrame(winners)

    print_section("How consistent is the winning choice across seeds")
    score_type_counts = winners_df["score_type"].value_counts()
    print("Score type win counts:")
    print(score_type_counts.to_string())

    combo_counts = winners_df.groupby(["score_type", "beta", "latent_dim"]).size().sort_values(ascending=False)
    print("\nFull (score_type, beta, latent_dim) win counts:")
    print(combo_counts.to_string())

    print_section(f"Production configuration check: {PRODUCTION_SCORE_TYPE}, "
                  f"beta={PRODUCTION_BETA}, latent_dim={PRODUCTION_LATENT_DIM}")
    prod_col = f"d_{PRODUCTION_SCORE_TYPE}_{s}degc"
    prod_rows = combined_df[(combined_df["beta"] == PRODUCTION_BETA) & (combined_df["latent_dim"] == PRODUCTION_LATENT_DIM)]
    print("Production configuration's KL effect size in each seed's own validation split:")
    for _, row in prod_rows.iterrows():
        is_winner = ((winners_df["seed"] == row["seed"]) & (winners_df["score_type"] == PRODUCTION_SCORE_TYPE) &
                     (winners_df["beta"] == PRODUCTION_BETA) & (winners_df["latent_dim"] == PRODUCTION_LATENT_DIM)).any()
        marker = " (won this seed)" if is_winner else ""
        print(f"  seed {int(row['seed'])}: d = {row[prod_col]:.2f}{marker}")

    mean_prod_effect = prod_rows[prod_col].mean()
    std_prod_effect = prod_rows[prod_col].std()
    print(f"\nProduction config across all seeds: mean d = {mean_prod_effect:.2f}, std = {std_prod_effect:.2f}")
    if std_prod_effect < mean_prod_effect * 0.3:
        print("Relatively stable across seeds -- the production configuration's discriminability "
              "does not appear to be an artifact of the original seed choice.")
    else:
        print("Notable variation across seeds -- the production configuration's discriminability "
              "is somewhat seed-dependent. Report this honestly rather than treating the single-seed "
              "result as a precise number.")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    score_type_counts.reindex(["kl", "combined", "recon"], fill_value=0).plot(
        kind="bar", ax=ax1, color=["#6699BF", "#8B5A2B", "#B23A48"]
    )
    ax1.set_ylabel(f"Number of seeds (of {len(SEEDS)}) where this score type won")
    ax1.set_title("Which anomaly score type wins, by seed")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)

    ax2.plot(prod_rows["seed"].astype(str), prod_rows[prod_col], marker="o", color="#6699BF")
    ax2.axhline(0.5, color="#B23A48", linestyle="--", linewidth=0.8)
    ax2.text(0, 0.55, "medium effect (d=0.5)", fontsize=8, color="#B23A48")
    ax2.set_xlabel("Seed")
    ax2.set_ylabel("KL effect size (Cohen's d)")
    ax2.set_title(f"Production config (beta={PRODUCTION_BETA}, latent={PRODUCTION_LATENT_DIM}) "
                  f"across seeds")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "vae_multiseed_robustness.png", dpi=150)
    print(f"\nSaved figure to {FIGURES_DIR / 'vae_multiseed_robustness.png'}")

    print_section("DONE")
    print("This is a robustness check, not a re-selection. The production model in models/ is unchanged. "
          "Decide separately whether these results change what should be reported as the primary "
          "configuration.")


if __name__ == "__main__":
    main()