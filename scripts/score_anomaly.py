"""
score_anomaly.py

The actual Method B result: scores 2026's pre-event spatial fields for
anomaly using the trained VAE, and reports where 2026 falls relative to
the historical (1950-2025) distribution of the same score -- structurally
the same question Method A asked, via a completely different mechanism.

Primary anomaly score: KL divergence (selected via compare_hyperparameters.py's
synthetic-anomaly discriminability test -- reconstruction error alone was a
much weaker discriminator at this site's realistic anomaly magnitude; see
vae_config.npz for which score type this model was chosen for). Reconstruction
error is still reported alongside it, for comparison and transparency, but
KL is the score actually used for the headline finding.

This is still attribution-adjacent screening, not formal detection-and-
attribution -- same framing as Method A, same site, same event, genuinely
different mechanism.

Requirements
------------
    pip install torch numpy matplotlib scipy
Must be run from the same scripts/ folder as train_vae.py (imports from it).
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy import stats as scipy_stats

from train_vae import ConvVAE, DEVICE

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "t2m_fields"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

EVENT_YEAR = 2026
TREND_WINDOW_DAYS = 30  # same lookback windows as Method A, for direct comparability
LOOKBACK_WINDOWS_DAYS = [7, 14, 30]


def print_section(title):
    print("\n" + "=" * 78)
    print(title)


def load_model_and_config():
    config = np.load(MODELS_DIR / "vae_config.npz", allow_pickle=True)
    latent_dim = int(config["latent_dim"])
    input_size = int(config["input_size"])
    anomaly_score_type = str(config["anomaly_score_type"]) if "anomaly_score_type" in config else "kl"
    training_years = set(int(y) for y in config["training_years"])
    validation_years = set(int(y) for y in config["validation_years"])

    model = ConvVAE(latent_dim=latent_dim, input_size=input_size).to(DEVICE)
    model.load_state_dict(torch.load(MODELS_DIR / "vae_best.pt", map_location=DEVICE))
    model.eval()

    norm_stats = np.load(MODELS_DIR / "normalization_stats.npz")
    pixel_mean, pixel_std = norm_stats["pixel_mean"], norm_stats["pixel_std"]

    print(f"Loaded model: latent_dim={latent_dim}, anomaly_score_type='{anomaly_score_type}'")
    print(f"Training years: {len(training_years)}, validation years: {len(validation_years)} "
          f"(validation years are included in the historical distribution below, same as "
          f"Method A's approach of using all non-event years as the reference distribution)")

    return model, anomaly_score_type, training_years, validation_years, pixel_mean, pixel_std


def compute_per_sample_scores(model, fields_norm):
    """Returns (recon_mse, kl, combined) per-sample arrays -- same definitions as compare_hyperparameters.py."""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(fields_norm).float().unsqueeze(1).to(DEVICE)
        recon, mu, logvar = model(x)
        recon_mse = ((recon - x) ** 2).mean(dim=[1, 2, 3]).cpu().numpy()
        kl = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)).cpu().numpy()
    return recon_mse, kl, recon_mse + kl


def load_all_years_fields():
    files = sorted(DATA_DIR.glob("t2m_fields_*.npz"))
    data_by_year = {}
    for f in files:
        year = int(f.stem.split("_")[-1])
        data = np.load(f)
        data_by_year[year] = data["t2m_degc"]  # shape (60, H, W): 60 days before Aug 26
    return data_by_year


def window_mean_score(daily_scores, window_days, total_days=60):
    """
    Mean score over the window_days immediately preceding the event (i.e.
    the LAST window_days entries of the 60-day fetched sequence), matching
    Method A's lookback-window definition for direct comparability.
    """
    return np.mean(daily_scores[total_days - window_days:])


def compute_percentile_rank(historical_values, test_value):
    """Same plotting-position formula used throughout this project: rank = count(<=)/(n+1)*100."""
    historical_values = np.asarray(historical_values)
    historical_values = historical_values[~np.isnan(historical_values)]
    n = len(historical_values)
    count_less_equal = np.sum(historical_values <= test_value)
    return (count_less_equal / (n + 1)) * 100, n


def score_all_years(model, data_by_year, pixel_mean, pixel_std):
    """
    For every year (including 2026), compute per-day recon/KL/combined
    scores, normalize inputs using the TRAINING-derived pixel_mean/std
    (2026 never influences normalization), then reduce to window-mean
    scores for each of the three lookback windows.
    """
    print_section("STEP 1: Scoring every year's pre-event fields")

    results = {"recon": {w: {} for w in LOOKBACK_WINDOWS_DAYS},
               "kl": {w: {} for w in LOOKBACK_WINDOWS_DAYS},
               "combined": {w: {} for w in LOOKBACK_WINDOWS_DAYS}}

    for year, fields_raw in data_by_year.items():
        fields_norm = np.nan_to_num((fields_raw - pixel_mean) / pixel_std, nan=0.0)
        recon_daily, kl_daily, combined_daily = compute_per_sample_scores(model, fields_norm)

        for window in LOOKBACK_WINDOWS_DAYS:
            results["recon"][window][year] = window_mean_score(recon_daily, window)
            results["kl"][window][year] = window_mean_score(kl_daily, window)
            results["combined"][window][year] = window_mean_score(combined_daily, window)

    print(f"Scored {len(data_by_year)} years across recon, KL, and combined scores, "
          f"{LOOKBACK_WINDOWS_DAYS} day windows.")
    return results


def report_percentile_ranks(results, anomaly_score_type):
    print_section("STEP 2: Percentile rank of 2026 vs. 1950-2025 (all three score types)")

    summary = {}
    for score_name in ["recon", "kl", "combined"]:
        summary[score_name] = {}
        for window in LOOKBACK_WINDOWS_DAYS:
            year_scores = results[score_name][window]
            event_value = year_scores[EVENT_YEAR]
            historical_values = [v for y, v in year_scores.items() if y != EVENT_YEAR]
            percentile, n_hist = compute_percentile_rank(historical_values, event_value)
            summary[score_name][window] = {"event_value": event_value, "percentile": percentile, "n_hist": n_hist}
            marker = "  <-- PRIMARY (selected via discriminability test)" if score_name == anomaly_score_type else ""
            print(f"  {score_name:9s} | {window:2d}-day window: 2026 = {event_value:.3f} | "
                  f"percentile = {percentile:5.1f} (n={n_hist}){marker}")

    return summary


def make_figure(results, summary, anomaly_score_type):
    print_section("STEP 3: Figure")

    primary_window = TREND_WINDOW_DAYS
    fig, axes = plt.subplots(1, len(LOOKBACK_WINDOWS_DAYS), figsize=(15, 5))

    for ax, window in zip(axes, LOOKBACK_WINDOWS_DAYS):
        year_scores = results[anomaly_score_type][window]
        years = sorted(y for y in year_scores if y != EVENT_YEAR)
        values = [year_scores[y] for y in years]
        event_value = year_scores[EVENT_YEAR]
        percentile = summary[anomaly_score_type][window]["percentile"]

        ax.hist(values, bins=20, color="#6699BF", alpha=0.7)
        ax.axvline(event_value, color="#B23A48", linewidth=2)
        ax.set_title(f"{window}-day window\n2026: p{percentile:.0f}", fontsize=11)
        ax.set_xlabel(f"{anomaly_score_type} score")
        ax.set_ylabel("count of historical years")

    fig.suptitle(f"Method B: VAE {anomaly_score_type} anomaly score, 2026 vs. 1950-2025", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "method_b_anomaly_score.png", dpi=150)
    print(f"Saved figure to {FIGURES_DIR / 'method_b_anomaly_score.png'}")


def save_results_csv(results):
    import pandas as pd
    rows = []
    for score_name in ["recon", "kl", "combined"]:
        for window in LOOKBACK_WINDOWS_DAYS:
            for year, value in results[score_name][window].items():
                rows.append({"score_type": score_name, "window_days": window, "year": year, "value": value})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "method_b_anomaly_scores.csv", index=False)
    print(f"Saved full per-year, per-window, per-score-type results to "
          f"{RESULTS_DIR / 'method_b_anomaly_scores.csv'}")


if __name__ == "__main__":
    model, anomaly_score_type, training_years, validation_years, pixel_mean, pixel_std = load_model_and_config()
    data_by_year = load_all_years_fields()

    if EVENT_YEAR not in data_by_year:
        raise FileNotFoundError(f"No fetched data for {EVENT_YEAR} -- check data/raw/t2m_fields/")

    results = score_all_years(model, data_by_year, pixel_mean, pixel_std)
    summary = report_percentile_ranks(results, anomaly_score_type)
    make_figure(results, summary, anomaly_score_type)
    save_results_csv(results)

    print_section("DONE")
    print(f"Primary result (KL score, matching Method A's window structure). "
          f"Compare this directly against Method A's percentile findings "
          f"(7-day p99, 14-day p95, 30-day p91) when writing this up.")