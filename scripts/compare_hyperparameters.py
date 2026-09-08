"""
compare_hyperparameters.py

Trains several (beta, latent_dim) combinations on the same train/validation
split as train_vae.py, and selects the best one by ANOMALY DISCRIMINABILITY
on synthetic anomalies -- not by validation reconstruction loss alone.

Why not just pick lowest validation reconstruction loss
---------------------------------------------------------
An earlier version of this script did exactly that, and the result exposed
the flaw directly: reconstruction loss decreased monotonically as beta
decreased and latent_dim increased, with no interior optimum -- the
"best" config was just whichever one sat at the edge of the grid searched.
Weakening the KL constraint and adding capacity always improves
reconstruction, because it makes the model behave more like a plain
autoencoder that can reconstruct nearly anything, including inputs it
has never seen. That is actively bad for anomaly detection: if the model
reconstructs an anomalous 2026 field just as well as a normal one, the
anomaly signal (reconstruction error) is uninformatively small precisely
because the model was selected to minimize it on ANY input.

What this version does instead
--------------------------------
For each trained configuration, synthetic anomalies are created by adding
a uniform temperature shift to held-out validation fields (2026 is never
touched during model selection). Reconstruction error is compared between
real validation fields and their shifted counterparts. The configuration
that best SEPARATES normal from shifted fields (largest effect size, not
smallest reconstruction error) is selected -- this directly tests the
property the model actually needs to have, rather than a proxy for it.

Shift magnitudes used (0.5, 1.0, 2.0 degC) span from realistic (Method A
found roughly a 0.6-1 degC pre-event anomaly at this site) to generous, so
the selected configuration isn't just tuned to one arbitrary magnitude.

Requirements
------------
    pip install torch numpy matplotlib pandas
Must be run from the same scripts/ folder as train_vae.py (imports from it).
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from train_vae import (
    ConvVAE, vae_loss, FieldsDataset, load_all_years, split_years, stack_years,
    DEVICE, RANDOM_SEED, N_VALIDATION_YEARS, BATCH_SIZE, LEARNING_RATE,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

BETA_VALUES = [0.1, 0.5, 1.0]
LATENT_DIMS = [8, 16, 32]
NUM_EPOCHS_SWEEP = 50
KL_COLLAPSE_THRESHOLD = 0.5
SYNTHETIC_SHIFT_MAGNITUDES_DEGC = [0.5, 1.0, 2.0]
SELECTION_SHIFT_MAGNITUDE = 1.0  # primary magnitude used to pick the winning config


def print_section(title):
    print("\n" + "=" * 78)
    print(title)


def train_one_config(latent_dim, beta, num_epochs, train_loader, val_loader, input_size):
    model = ConvVAE(latent_dim=latent_dim, input_size=input_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = {"train_loss": [], "val_loss": [], "train_recon": [], "val_recon": [], "train_kl": [], "val_kl": []}

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses, train_recons, train_kls = [], [], []
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss, recon_loss, kl_loss = vae_loss(recon, batch, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_recons.append(recon_loss.item())
            train_kls.append(kl_loss.item())

        model.eval()
        val_losses, val_recons, val_kls = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                recon, mu, logvar = model(batch)
                loss, recon_loss, kl_loss = vae_loss(recon, batch, mu, logvar, beta)
                val_losses.append(loss.item())
                val_recons.append(recon_loss.item())
                val_kls.append(kl_loss.item())

        history["train_loss"].append(np.mean(train_losses))
        history["val_loss"].append(np.mean(val_losses))
        history["train_recon"].append(np.mean(train_recons))
        history["val_recon"].append(np.mean(val_recons))
        history["train_kl"].append(np.mean(train_kls))
        history["val_kl"].append(np.mean(val_kls))

    return model, history


def compute_per_sample_recon_mse(model, fields_norm):
    """Per-sample reconstruction MSE (mean over pixels), for a normalized (N, H, W) array."""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(fields_norm).float().unsqueeze(1).to(DEVICE)
        recon, mu, logvar = model(x)
        per_sample_mse = ((recon - x) ** 2).mean(dim=[1, 2, 3]).cpu().numpy()
    return per_sample_mse


def compute_per_sample_scores(model, fields_norm):
    """
    Returns three per-sample anomaly scores for a normalized (N, H, W) array:
    reconstruction MSE, KL divergence, and their unweighted sum (a
    beta-free combined score -- beta controls training regularization, but
    the score used FOR anomaly detection doesn't need to inherit that same
    weighting).
    """
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(fields_norm).float().unsqueeze(1).to(DEVICE)
        recon, mu, logvar = model(x)
        recon_mse = ((recon - x) ** 2).mean(dim=[1, 2, 3]).cpu().numpy()
        kl = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)).cpu().numpy()
    combined = recon_mse + kl
    return recon_mse, kl, combined


def effect_size(anomalous_scores, normal_scores):
    """
    Cohen's d: how many standard deviations of the NORMAL distribution
    separate the two groups' means. This is the actual quantity we care
    about -- not which config has the lowest absolute reconstruction error,
    but which config's error distribution shifts the most, relative to its
    own spread, when the input is genuinely anomalous.
    """
    pooled_std = normal_scores.std()
    if pooled_std < 1e-8:
        return 0.0
    return (anomalous_scores.mean() - normal_scores.mean()) / pooled_std


def run_sweep():
    print_section("STEP 1: Loading data (same split as train_vae.py)")
    data_by_year = load_all_years()
    training_years, validation_years = split_years(data_by_year, N_VALIDATION_YEARS, RANDOM_SEED)
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

    print_section(f"STEP 2: Sweeping {len(BETA_VALUES)} x {len(LATENT_DIMS)} = "
                  f"{len(BETA_VALUES) * len(LATENT_DIMS)} configurations, {NUM_EPOCHS_SWEEP} epochs each")
    print(f"Device: {DEVICE}")

    trained_models = {}
    final_metrics = {}
    start_time = time.time()

    for beta in BETA_VALUES:
        for latent_dim in LATENT_DIMS:
            print(f"\n  Training beta={beta}, latent_dim={latent_dim}...")
            model, history = train_one_config(latent_dim, beta, NUM_EPOCHS_SWEEP, train_loader, val_loader, input_size)
            final_val_recon = np.mean(history["val_recon"][-5:])
            final_val_kl = np.mean(history["val_kl"][-5:])
            print(f"    val_recon={final_val_recon:.3f} | val_kl={final_val_kl:.3f}")
            trained_models[(beta, latent_dim)] = model
            final_metrics[(beta, latent_dim)] = {"val_recon": final_val_recon, "val_kl": final_val_kl}

    elapsed_minutes = (time.time() - start_time) / 60
    print(f"\nSweep complete. Elapsed: {elapsed_minutes:.1f} minutes.")

    return trained_models, final_metrics, training_years, validation_years, val_fields_raw, val_fields_norm, pixel_mean, pixel_std, input_size


def evaluate_discriminability(trained_models, final_metrics, val_fields_raw, val_fields_norm, pixel_mean, pixel_std):
    """
    For each trained config, compute the effect size (Cohen's d) separating
    real validation fields from synthetically-shifted versions of the same
    fields, at each shift magnitude -- for THREE candidate anomaly scores:
    reconstruction error alone, KL divergence alone, and their sum. The
    original Method B plan specified both signals; testing only
    reconstruction error would risk concluding the method is weak when the
    gap might actually be in which score is used, not the model itself.
    """
    print_section("STEP 3: Anomaly discriminability (synthetic shift injection)")

    results = []
    for (beta, latent_dim), model in trained_models.items():
        normal_recon, normal_kl, normal_combined = compute_per_sample_scores(model, val_fields_norm)
        collapsed = final_metrics[(beta, latent_dim)]["val_kl"] < KL_COLLAPSE_THRESHOLD

        row = {"beta": beta, "latent_dim": latent_dim,
               "val_recon": final_metrics[(beta, latent_dim)]["val_recon"],
               "val_kl": final_metrics[(beta, latent_dim)]["val_kl"],
               "posterior_collapse_flag": collapsed}

        for shift in SYNTHETIC_SHIFT_MAGNITUDES_DEGC:
            shifted_raw = val_fields_raw + shift
            shifted_norm = np.nan_to_num((shifted_raw - pixel_mean) / pixel_std, nan=0.0)
            anom_recon, anom_kl, anom_combined = compute_per_sample_scores(model, shifted_norm)
            row[f"d_recon_{shift}degc"] = effect_size(anom_recon, normal_recon)
            row[f"d_kl_{shift}degc"] = effect_size(anom_kl, normal_kl)
            row[f"d_combined_{shift}degc"] = effect_size(anom_combined, normal_combined)

        s = SELECTION_SHIFT_MAGNITUDE
        print(f"  beta={beta}, latent_dim={latent_dim} @ {s}degC: "
              f"d_recon={row[f'd_recon_{s}degc']:.2f} | d_kl={row[f'd_kl_{s}degc']:.2f} | "
              f"d_combined={row[f'd_combined_{s}degc']:.2f}" +
              (" *** POSSIBLE POSTERIOR COLLAPSE ***" if collapsed else ""))

        results.append(row)

    return pd.DataFrame(results)


def summarize_and_select(results_df, trained_models, training_years, validation_years, pixel_mean, pixel_std, input_size):
    print_section("STEP 4: Selecting best configuration and best score type")

    s = SELECTION_SHIFT_MAGNITUDE
    score_cols = [f"d_recon_{s}degc", f"d_kl_{s}degc", f"d_combined_{s}degc"]
    non_collapsed = results_df[~results_df["posterior_collapse_flag"]].copy()
    if non_collapsed.empty:
        print("\nWARNING: every configuration showed possible posterior collapse. Try lower beta and re-run.")
        return

    # Which SCORE TYPE (recon, kl, combined) achieves the best separation
    # anywhere in the grid -- decide this before picking a config, since a
    # config that looks mediocre on recon alone might be the best performer
    # once KL or the combined score is considered.
    best_per_score_type = {col: non_collapsed.loc[non_collapsed[col].idxmax()] for col in score_cols}
    print(f"Best result per score type, at {s} degC shift:")
    for col, row in best_per_score_type.items():
        print(f"  {col}: beta={row['beta']}, latent_dim={int(row['latent_dim'])}, d={row[col]:.2f}")

    best_score_col = max(score_cols, key=lambda c: best_per_score_type[c][c])
    best_row = best_per_score_type[best_score_col]
    best_beta, best_latent_dim = best_row["beta"], int(best_row["latent_dim"])
    score_type_label = best_score_col.replace(f"_{s}degc", "").replace("d_", "")

    print(f"\nBest overall: {score_type_label} score, beta={best_beta}, latent_dim={best_latent_dim} "
          f"(effect size = {best_row[best_score_col]:.2f} standard deviations at {s} degC shift)")

    if best_row[best_score_col] < 0.5:
        print(f"\nCAUTION: even the best score/config combination only reaches a SMALL effect size "
              f"(Cohen's d = {best_row[best_score_col]:.2f}) at the realistic {s} degC shift magnitude "
              f"Method A actually found. This should be stated plainly in the writeup, not smoothed "
              f"over -- Method B's anomaly-detection power at this site, at this magnitude, may be "
              f"genuinely limited, which is itself a legitimate methodological finding: it would mean "
              f"the classical statistics in Method A were the more sensitive detector for this "
              f"particular anomaly, not merely a simpler one.")

    results_df.to_csv(RESULTS_DIR / "vae_hyperparameter_sweep.csv", index=False)
    print(f"\nSaved full comparison table (all score types, all magnitudes) to "
          f"{RESULTS_DIR / 'vae_hyperparameter_sweep.csv'}")

    # Plot: for each score type, latent_dim vs effect size at the selection
    # magnitude, one line per beta -- shows not just which config wins, but
    # how much the choice of SCORE TYPE itself matters.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, col, label in zip(axes, score_cols, ["Reconstruction error", "KL divergence", "Combined (recon + KL)"]):
        for beta in BETA_VALUES:
            subset = results_df[results_df["beta"] == beta].sort_values("latent_dim")
            ax.plot(subset["latent_dim"], subset[col], marker="o", label=f"beta={beta}")
        ax.set_xlabel("Latent dimension")
        ax.set_title(label)
        ax.axhline(0, color="#999999", linewidth=0.7)
        ax.axhline(0.5, color="#B23A48", linewidth=0.7, linestyle="--")
    axes[0].text(LATENT_DIMS[0], 0.55, "medium effect (d=0.5)", fontsize=7.5, color="#B23A48")
    axes[0].set_ylabel(f"Effect size (Cohen's d) at +{s}\u00b0C shift")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Anomaly discriminability by score type and configuration ({s}\u00b0C synthetic shift)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "vae_hyperparameter_sweep.png", dpi=150)
    print(f"Saved comparison plot to {FIGURES_DIR / 'vae_hyperparameter_sweep.png'}")

    best_model = trained_models[(best_beta, best_latent_dim)]
    torch.save(best_model.state_dict(), MODELS_DIR / "vae_best.pt")
    torch.save(best_model.state_dict(), MODELS_DIR / "vae_final.pt")
    np.savez_compressed(MODELS_DIR / "normalization_stats.npz", pixel_mean=pixel_mean, pixel_std=pixel_std)
    np.savez(MODELS_DIR / "vae_config.npz",
              latent_dim=np.array(best_latent_dim), input_size=np.array(input_size),
              beta=np.array(best_beta), anomaly_score_type=np.array(score_type_label),
              training_years=np.array(training_years), validation_years=np.array(validation_years))
    print(f"\nSaved the best configuration's weights to {MODELS_DIR / 'vae_best.pt'} (and vae_final.pt), "
          f"with anomaly_score_type='{score_type_label}' recorded in vae_config.npz so score_anomaly.py "
          f"uses the same score type this was selected on. "
          f"NOTE: trained for only {NUM_EPOCHS_SWEEP} epochs, not train_vae.py's full 100 -- "
          f"consider re-running train_vae.py with BETA={best_beta} and LATENT_DIM={best_latent_dim} "
          f"for the full epoch count before treating this as final.")


if __name__ == "__main__":
    trained_models, final_metrics, training_years, validation_years, val_fields_raw, val_fields_norm, pixel_mean, pixel_std, input_size = run_sweep()
    results_df = evaluate_discriminability(trained_models, final_metrics, val_fields_raw, val_fields_norm, pixel_mean, pixel_std)
    summarize_and_select(results_df, trained_models, training_years, validation_years, pixel_mean, pixel_std, input_size)

    print_section("DONE")
    print("Sweep complete. Review the comparison plot before deciding whether to re-run "
          "train_vae.py at full epochs with the selected configuration, or proceed to "
          "score_anomaly.py as-is.")