"""
diagnose_and_explain.py

Two things, both aimed at the 7-day discrepancy between Method A (p99) and
Method B's KL score (p71) for 2026:

  1. A raw diagnostic, independent of the VAE entirely: the actual spatial
     temperature field for 2026's last 7 pre-event days, compared against
     the climatological mean field for the same 7-day window across
     1950-2025. This answers "where was the warmth, spatially" using
     nothing but the data itself.

  2. SHAP explainability for the VAE's KL anomaly score: which pixels in
     the input field actually drive the model's KL output. This is the
     model-grounded version of the same question -- not just "where is it
     warm" but "where does THIS MODEL'S anomaly signal actually come from."

Together these test a specific hypothesis: that the last-week warmth was a
sharp, localized spike at the exact site coordinate, without the broader
~220 km region's spatial PATTERN looking unusual -- which would explain why
a point-based method (Method A) caught it sharply while a regional
spatial-field method (Method B) did not. If the SHAP attribution is
concentrated near the site pixel specifically, and the raw difference map
also shows a localized rather than region-wide anomaly, that supports this
explanation. If SHAP attribution is diffuse or concentrated elsewhere, this
should be reported honestly as an open question, not forced into the
tidiest available story.

KL divergence depends only on the encoder (mu, logvar) -- not decoding --
so the wrapper module used for SHAP skips reconstruction entirely.

Requirements
------------
    pip install torch numpy matplotlib shap
Must be run from the same scripts/ folder as train_vae.py (imports from it).
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import shap

from train_vae import ConvVAE, DEVICE

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "t2m_fields"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

EVENT_YEAR = 2026
DISCREPANCY_WINDOW_DAYS = 7
TOTAL_FETCHED_DAYS = 60
N_SHAP_BACKGROUND_SAMPLES = 200
RANDOM_SEED = 42


def print_section(title):
    print("\n" + "=" * 78)
    print(title)


def load_model_and_config():
    config = np.load(MODELS_DIR / "vae_config.npz", allow_pickle=True)
    latent_dim = int(config["latent_dim"])
    input_size = int(config["input_size"])
    training_years = set(int(y) for y in config["training_years"])

    model = ConvVAE(latent_dim=latent_dim, input_size=input_size).to(DEVICE)
    model.load_state_dict(torch.load(MODELS_DIR / "vae_best.pt", map_location=DEVICE))
    model.eval()

    norm_stats = np.load(MODELS_DIR / "normalization_stats.npz")
    return model, training_years, norm_stats["pixel_mean"], norm_stats["pixel_std"]


def load_all_years_fields():
    files = sorted(DATA_DIR.glob("t2m_fields_*.npz"))
    data_by_year = {}
    for f in files:
        year = int(f.stem.split("_")[-1])
        data_by_year[year] = np.load(f)["t2m_degc"]  # shape (60, H, W)
    return data_by_year


def compute_per_day_kl(model, fields_norm):
    """Per-day KL divergence for a (n_days, H, W) normalized array."""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(fields_norm).float().unsqueeze(1).to(DEVICE)
        mu, logvar = model.encode(x)
        kl = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)).cpu().numpy()
    return kl


# ---------------------------------------------------------------------------
# Part 1: raw field diagnostic (no model involved)
# ---------------------------------------------------------------------------
def raw_field_diagnostic(data_by_year, window_days):
    print_section("PART 1: Raw spatial field diagnostic (no model, just the data)")

    event_last_days = data_by_year[EVENT_YEAR][TOTAL_FETCHED_DAYS - window_days:]
    event_mean_field = np.nanmean(event_last_days, axis=0)

    historical_fields = []
    for year, fields in data_by_year.items():
        if year == EVENT_YEAR:
            continue
        historical_fields.append(fields[TOTAL_FETCHED_DAYS - window_days:])
    historical_stack = np.concatenate(historical_fields, axis=0)
    climatology_mean_field = np.nanmean(historical_stack, axis=0)

    difference_field = event_mean_field - climatology_mean_field

    print(f"2026 mean field range: {event_mean_field.min():.2f} to {event_mean_field.max():.2f} degC")
    print(f"Climatology mean field range: {climatology_mean_field.min():.2f} to {climatology_mean_field.max():.2f} degC")
    print(f"Difference field range: {difference_field.min():.2f} to {difference_field.max():.2f} degC")

    center_idx = difference_field.shape[0] // 2
    center_region = difference_field[center_idx - 2:center_idx + 3, center_idx - 2:center_idx + 3]
    print(f"Difference near the site (5x5 pixels around center): mean={center_region.mean():.2f} degC, "
          f"vs. whole-field mean={difference_field.mean():.2f} degC")
    if abs(center_region.mean()) > abs(difference_field.mean()) * 1.5:
        print("The anomaly is CONCENTRATED near the site relative to the whole field -- consistent with "
              "a localized rather than region-wide warm spike.")
    else:
        print("The anomaly near the site is NOT much stronger than the field-wide average -- "
              "the warmth (or lack thereof) looks fairly region-wide, not concentrated at the site.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    im0 = axes[0].imshow(event_mean_field, cmap="RdBu_r")
    axes[0].set_title(f"2026: mean field,\nlast {window_days} days")
    plt.colorbar(im0, ax=axes[0], label="degC", fraction=0.046)

    im1 = axes[1].imshow(climatology_mean_field, cmap="RdBu_r")
    axes[1].set_title(f"1950-2025 climatology,\nsame {window_days}-day window")
    plt.colorbar(im1, ax=axes[1], label="degC", fraction=0.046)

    max_abs_diff = np.abs(difference_field).max()
    im2 = axes[2].imshow(difference_field, cmap="RdBu_r", vmin=-max_abs_diff, vmax=max_abs_diff)
    axes[2].set_title(f"Difference\n(2026 minus climatology)")
    plt.colorbar(im2, ax=axes[2], label="degC", fraction=0.046)

    for ax in axes:
        ax.scatter([ax.get_images()[0].get_array().shape[1] // 2],
                   [ax.get_images()[0].get_array().shape[0] // 2],
                   marker="x", color="black", s=80)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Raw spatial diagnostic: {window_days}-day pre-event field vs. climatology "
                 f"(x marks the source-zone site)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "diagnostic_raw_field_comparison.png", dpi=150)
    print(f"Saved to {FIGURES_DIR / 'diagnostic_raw_field_comparison.png'}")


# ---------------------------------------------------------------------------
# Part 2: SHAP explainability for the KL anomaly score
# ---------------------------------------------------------------------------
class KLScoreModule(nn.Module):
    """Wraps the VAE encoder to output per-sample KL divergence as a (batch, 1) tensor -- the actual
    anomaly score used in score_anomaly.py. Decoding is skipped entirely since KL doesn't need it."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, x):
        mu, logvar = self.vae.encode(x)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1, keepdim=True)
        return kl


def shap_explain_last_week(model, data_by_year, training_years, pixel_mean, pixel_std, window_days):
    print_section("PART 2: SHAP explainability for the KL anomaly score")

    kl_module = KLScoreModule(model).to(DEVICE)
    kl_module.eval()

    rng = np.random.default_rng(RANDOM_SEED)
    background_fields = []
    for year in training_years:
        background_fields.append(data_by_year[year])
    background_stack = np.concatenate(background_fields, axis=0)
    background_norm = np.nan_to_num((background_stack - pixel_mean) / pixel_std, nan=0.0)
    background_idx = rng.choice(len(background_norm), size=min(N_SHAP_BACKGROUND_SAMPLES, len(background_norm)), replace=False)
    background_sample = background_norm[background_idx]
    background_tensor = torch.from_numpy(background_sample).float().unsqueeze(1).to(DEVICE)
    print(f"Background: {len(background_sample)} randomly sampled days from training years.")

    event_last_days_raw = data_by_year[EVENT_YEAR][TOTAL_FETCHED_DAYS - window_days:]
    event_last_days_norm = np.nan_to_num((event_last_days_raw - pixel_mean) / pixel_std, nan=0.0)
    explain_tensor = torch.from_numpy(event_last_days_norm).float().unsqueeze(1).to(DEVICE)

    print(f"Explaining the last {window_days} days of 2026 (the discrepancy window)...")
    explainer = shap.GradientExplainer(kl_module, background_tensor)
    shap_values = explainer.shap_values(explain_tensor)

    # shap_values shape: (n_days, 1, H, W, 1) for a single-output model in
    # recent shap versions, or (n_days, 1, H, W) in older ones -- normalize
    # to (n_days, H, W) regardless.
    shap_values = np.array(shap_values)
    shap_values = shap_values.reshape(window_days, explain_tensor.shape[-2], explain_tensor.shape[-1])

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    center_idx = mean_abs_shap.shape[0] // 2
    center_region = mean_abs_shap[center_idx - 2:center_idx + 3, center_idx - 2:center_idx + 3]
    print(f"Mean |SHAP value| near the site (5x5 pixels): {center_region.mean():.5f}")
    print(f"Mean |SHAP value| across the whole field: {mean_abs_shap.mean():.5f}")
    if center_region.mean() > mean_abs_shap.mean() * 1.5:
        print("The model's KL score is driven PRIMARILY by pixels near the site -- consistent with the "
              "model picking up a localized signal, even if it didn't rank the week as strongly anomalous.")
    else:
        print("The model's KL score draws on the WHOLE region fairly evenly, not concentrated at the site -- "
              "this is a genuine property of how the model represents these fields, not an artifact of "
              "the anomaly being missed only at the site pixel.")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i in range(window_days):
        ax = axes.flatten()[i]
        max_abs = np.abs(shap_values[i]).max()
        im = ax.imshow(shap_values[i], cmap="RdBu_r", vmin=-max_abs, vmax=max_abs)
        ax.set_title(f"Day -{window_days - i}", fontsize=10)
        ax.scatter([shap_values[i].shape[1] // 2], [shap_values[i].shape[0] // 2], marker="x", color="black", s=50)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)

    ax_agg = axes.flatten()[7]
    max_abs = mean_abs_shap.max()
    im = ax_agg.imshow(mean_abs_shap, cmap="Reds", vmin=0, vmax=max_abs)
    ax_agg.set_title("Mean |SHAP|\n(whole week)", fontsize=10, fontweight="bold")
    ax_agg.scatter([mean_abs_shap.shape[1] // 2], [mean_abs_shap.shape[0] // 2], marker="x", color="black", s=50)
    ax_agg.set_xticks([])
    ax_agg.set_yticks([])
    plt.colorbar(im, ax=ax_agg, fraction=0.046)

    fig.suptitle(f"SHAP attribution for the KL anomaly score, last {window_days} days of 2026 "
                 f"(x marks the source-zone site)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "diagnostic_shap_attribution.png", dpi=150)
    print(f"Saved to {FIGURES_DIR / 'diagnostic_shap_attribution.png'}")


# ---------------------------------------------------------------------------
# Part 3: day-by-day KL time series for context
# ---------------------------------------------------------------------------
def daily_kl_timeseries(model, data_by_year, pixel_mean, pixel_std):
    print_section("PART 3: Day-by-day KL score, last 30 days of 2026 vs. historical range")

    event_fields_norm = np.nan_to_num((data_by_year[EVENT_YEAR][-30:] - pixel_mean) / pixel_std, nan=0.0)
    event_daily_kl = compute_per_day_kl(model, event_fields_norm)

    historical_daily_kl = []
    for year, fields in data_by_year.items():
        if year == EVENT_YEAR:
            continue
        fields_norm = np.nan_to_num((fields[-30:] - pixel_mean) / pixel_std, nan=0.0)
        historical_daily_kl.append(compute_per_day_kl(model, fields_norm))
    historical_stack = np.stack(historical_daily_kl, axis=0)  # (n_years, 30)

    p10 = np.percentile(historical_stack, 10, axis=0)
    p50 = np.percentile(historical_stack, 50, axis=0)
    p90 = np.percentile(historical_stack, 90, axis=0)

    days_before_event = np.arange(30, 0, -1)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(days_before_event, p10, p90, color="#6699BF", alpha=0.3, label="Historical 10th-90th percentile")
    ax.plot(days_before_event, p50, color="#6699BF", linewidth=1.5, label="Historical median")
    ax.plot(days_before_event, event_daily_kl, color="#B23A48", linewidth=2, marker="o", markersize=4, label="2026")
    ax.axvspan(0.5, DISCREPANCY_WINDOW_DAYS + 0.5, color="#B23A48", alpha=0.08)
    ax.text(DISCREPANCY_WINDOW_DAYS, ax.get_ylim()[1] * 0.95, "7-day\nwindow", fontsize=8, color="#B23A48", ha="center", va="top")
    ax.invert_xaxis()
    ax.set_xlabel("Days before Aug 26")
    ax.set_ylabel("KL score")
    ax.set_title("Day-by-day KL score: 2026 vs. historical range")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "diagnostic_daily_kl_timeseries.png", dpi=150)
    print(f"Saved to {FIGURES_DIR / 'diagnostic_daily_kl_timeseries.png'}")

    last_7 = event_daily_kl[-DISCREPANCY_WINDOW_DAYS:]
    print(f"2026's last {DISCREPANCY_WINDOW_DAYS} daily KL values: {np.round(last_7, 2).tolist()}")
    print(f"Std dev within this window: {last_7.std():.2f} -- "
          f"{'high day-to-day variance, one or two days may be dominating the average' if last_7.std() > last_7.mean() * 0.3 else 'fairly stable across the week, not a single-day artifact'}")


if __name__ == "__main__":
    model, training_years, pixel_mean, pixel_std = load_model_and_config()
    data_by_year = load_all_years_fields()

    raw_field_diagnostic(data_by_year, DISCREPANCY_WINDOW_DAYS)
    daily_kl_timeseries(model, data_by_year, pixel_mean, pixel_std)
    shap_explain_last_week(model, data_by_year, training_years, pixel_mean, pixel_std, DISCREPANCY_WINDOW_DAYS)

    print_section("DONE")
    print("Diagnostics complete. Review all three figures before writing up the 7-day "
          "discrepancy finding.")