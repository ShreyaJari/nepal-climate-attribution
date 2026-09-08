"""
train_vae.py

Trains a small convolutional variational autoencoder (VAE) on ERA5-Land
2m temperature spatial fields, learning what a "normal" pre-event thermal
pattern looks like at this site. Reconstruction error and KL divergence
for 2026's pre-event window (computed separately, in score_anomaly.py) then
serve as a threshold-free anomaly signal -- a genuinely different
methodology from Method A's percentile/trend statistics, applied to the
same underlying question.

All hyperparameters are configurable constants below, by design -- the
plan is to train once with reasonable defaults, inspect the results, then
adjust (latent size, beta, epochs, split) rather than treat this run as
final.

Data handling
-------------
  - 2026 is excluded entirely from training and validation -- it is the
    inference target, not training data.
  - The remaining 76 years are split BY YEAR, not by day. Splitting by
    individual day would leak information: consecutive days within the
    same 60-day window are strongly autocorrelated, so a day-level split
    would let the model see near-duplicates of "held-out" days during
    training.
  - Per-pixel normalization (mean and std maps, shape 21x21) is computed
    from the training years ONLY, then applied identically to validation
    and (later, in score_anomaly.py) to 2026 -- held-out data never
    influences the normalization statistics.

Requirements
------------
    pip install torch numpy matplotlib
"""

import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "t2m_fields"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

EVENT_YEAR = 2026

# --- Hyperparameters: adjust these and re-run, rather than editing logic below ---
LATENT_DIM = 8
NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
BETA = 0.5                 # weight on the KL divergence term. Selected via compare_hyperparameters.py's
                            # synthetic-anomaly discriminability sweep -- NOT the default beta=1.0 a
                            # standard VAE recipe would suggest. At beta=1.0, KL-based anomaly
                            # discriminability was measurably weaker (d=0.80 vs 0.96 at this site's
                            # realistic +1degC anomaly magnitude); beta=0.5 gave the best-separating
                            # latent space for THIS downstream task, which is a different optimization
                            # target than "best ELBO" or "best reconstruction."
ANOMALY_SCORE_TYPE = "kl"  # Selected via the same sweep: KL divergence alone discriminates the
                            # synthetic anomaly far better than reconstruction error (d=0.96 vs 0.36
                            # at latent_dim=8) -- the model doesn't need to reconstruct an anomalous
                            # field badly for it to map to an atypical region of latent space. Recorded
                            # here so score_anomaly.py uses the same score type this model was chosen for.
N_VALIDATION_YEARS = 8    # held out as whole years, not days
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available()
                       else "cpu")


def print_section(title):
    print("\n" + "=" * 78)
    print(title)


# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------
def load_all_years():
    """
    Load every fetched year's .npz except the event year, returning a dict
    of {year: (dates_array, fields_array)}.
    """
    files = sorted(DATA_DIR.glob("t2m_fields_*.npz"))
    data_by_year = {}
    for f in files:
        year = int(f.stem.split("_")[-1])
        if year == EVENT_YEAR:
            continue
        data = np.load(f)
        data_by_year[year] = (data["dates"], data["t2m_degc"])
    return data_by_year


def split_years(data_by_year, n_validation_years, seed):
    """
    Randomly select n_validation_years whole years for validation; the rest
    are training years. Uses a fixed seed for reproducibility, and prints
    the exact split so it's auditable, not just reproducible in principle.
    """
    years = sorted(data_by_year.keys())
    rng = random.Random(seed)
    validation_years = sorted(rng.sample(years, n_validation_years))
    training_years = sorted(set(years) - set(validation_years))
    print(f"Training years (n={len(training_years)}): {training_years}")
    print(f"Validation years (n={len(validation_years)}): {validation_years}")
    return training_years, validation_years


def stack_years(data_by_year, years):
    """Concatenate all days across the given years into one (N, H, W) array."""
    arrays = [data_by_year[y][1] for y in years]
    return np.concatenate(arrays, axis=0)


class FieldsDataset(Dataset):
    """Wraps a (N, H, W) array of normalized fields as a torch Dataset with a channel dim."""

    def __init__(self, fields_normalized):
        self.fields = torch.from_numpy(fields_normalized).float().unsqueeze(1)  # (N, 1, H, W)

    def __len__(self):
        return len(self.fields)

    def __getitem__(self, idx):
        return self.fields[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ConvVAE(nn.Module):
    """
    Small convolutional VAE for 21x21 single-channel temperature fields.
    Deliberately small (2 conv layers, latent_dim ~16): with ~4,500 training
    examples of a smooth, low-complexity signal, a larger network risks
    overfitting rather than learning anything more useful.

    Encoder: 21x21 -> 11x11 -> 6x6, flattened to fc_mu/fc_logvar.
    Decoder: mirrors the encoder exactly (6x6 -> 11x11 -> 21x21 via
    transposed convolutions with matching stride/padding), with a final
    interpolation to the exact target size as a safety net in case the
    input field size ever changes and the transpose-conv arithmetic no
    longer lines up exactly.
    """

    def __init__(self, latent_dim=LATENT_DIM, input_size=21):
        super().__init__()
        self.input_size = input_size

        self.enc_conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)   # 21 -> 11
        self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # 11 -> 6
        self.flatten_size = 32 * 6 * 6
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, self.flatten_size)
        self.dec_conv1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1)  # 6 -> 11
        self.dec_conv2 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1)   # 11 -> 21

    def encode(self, x):
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = h.flatten(start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(-1, 32, 6, 6)
        h = F.relu(self.dec_conv1(h))
        h = self.dec_conv2(h)
        if h.shape[-2:] != (self.input_size, self.input_size):
            h = F.interpolate(h, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        return h

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta):
    """Standard VAE loss: reconstruction MSE (summed over pixels) + beta * KL divergence, both averaged over the batch."""
    batch_size = x.shape[0]
    recon_loss = F.mse_loss(recon_x, x, reduction="sum") / batch_size
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train():
    print_section("STEP 1: Loading data")
    data_by_year = load_all_years()
    print(f"Loaded {len(data_by_year)} years (2026 excluded -- inference target, not training data).")

    training_years, validation_years = split_years(data_by_year, N_VALIDATION_YEARS, RANDOM_SEED)
    train_fields_raw = stack_years(data_by_year, training_years)
    val_fields_raw = stack_years(data_by_year, validation_years)
    print(f"Training set: {train_fields_raw.shape[0]} daily fields. Validation set: {val_fields_raw.shape[0]} daily fields.")

    print_section("STEP 2: Normalizing (stats computed from training years only)")
    pixel_mean = np.nanmean(train_fields_raw, axis=0)
    pixel_std = np.nanstd(train_fields_raw, axis=0)
    pixel_std = np.where(pixel_std < 1e-6, 1e-6, pixel_std)  # guard against divide-by-zero

    train_fields_norm = np.nan_to_num((train_fields_raw - pixel_mean) / pixel_std, nan=0.0)
    val_fields_norm = np.nan_to_num((val_fields_raw - pixel_mean) / pixel_std, nan=0.0)

    np.savez_compressed(MODELS_DIR / "normalization_stats.npz", pixel_mean=pixel_mean, pixel_std=pixel_std)
    print(f"Saved normalization stats to {MODELS_DIR / 'normalization_stats.npz'}")

    train_loader = DataLoader(FieldsDataset(train_fields_norm), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(FieldsDataset(val_fields_norm), batch_size=BATCH_SIZE, shuffle=False)

    print_section(f"STEP 3: Training on device: {DEVICE}")
    model = ConvVAE(latent_dim=LATENT_DIM, input_size=train_fields_raw.shape[-1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {"train_loss": [], "val_loss": [], "train_recon": [], "val_recon": [], "train_kl": [], "val_kl": []}
    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_losses, train_recons, train_kls = [], [], []
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss, recon_loss, kl_loss = vae_loss(recon, batch, mu, logvar, BETA)
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
                loss, recon_loss, kl_loss = vae_loss(recon, batch, mu, logvar, BETA)
                val_losses.append(loss.item())
                val_recons.append(recon_loss.item())
                val_kls.append(kl_loss.item())

        train_loss_mean = np.mean(train_losses)
        val_loss_mean = np.mean(val_losses)
        history["train_loss"].append(train_loss_mean)
        history["val_loss"].append(val_loss_mean)
        history["train_recon"].append(np.mean(train_recons))
        history["val_recon"].append(np.mean(val_recons))
        history["train_kl"].append(np.mean(train_kls))
        history["val_kl"].append(np.mean(val_kls))

        if val_loss_mean < best_val_loss:
            best_val_loss = val_loss_mean
            torch.save(model.state_dict(), MODELS_DIR / "vae_best.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{NUM_EPOCHS}: train_loss={train_loss_mean:.3f} "
                  f"(recon={history['train_recon'][-1]:.3f}, kl={history['train_kl'][-1]:.3f}) | "
                  f"val_loss={val_loss_mean:.3f} (recon={history['val_recon'][-1]:.3f}, kl={history['val_kl'][-1]:.3f})")

    torch.save(model.state_dict(), MODELS_DIR / "vae_final.pt")
    print(f"\nSaved final-epoch weights to {MODELS_DIR / 'vae_final.pt'} "
          f"and best-validation-loss weights to {MODELS_DIR / 'vae_best.pt'} (best val_loss={best_val_loss:.3f}).")

    # Save the config alongside the weights so score_anomaly.py can load the
    # right architecture and anomaly score type without hand-copying
    # hyperparameters. Matches the field names compare_hyperparameters.py
    # saves, so score_anomaly.py can read either interchangeably.
    config = {"latent_dim": LATENT_DIM, "input_size": int(train_fields_raw.shape[-1]),
              "beta": BETA, "anomaly_score_type": ANOMALY_SCORE_TYPE,
              "training_years": training_years, "validation_years": validation_years}
    np.savez(MODELS_DIR / "vae_config.npz", **{k: np.array(v) for k, v in config.items()})

    print_section("STEP 4: Loss curve")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = range(1, NUM_EPOCHS + 1)
    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Total loss (recon + beta*KL)")
    ax1.set_title("Total VAE loss")
    ax1.legend()

    ax2.plot(epochs, history["train_recon"], label="Train recon", color="#6699BF")
    ax2.plot(epochs, history["val_recon"], label="Val recon", color="#6699BF", linestyle="--")
    ax2.plot(epochs, history["train_kl"], label="Train KL", color="#B23A48")
    ax2.plot(epochs, history["val_kl"], label="Val KL", color="#B23A48", linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss component")
    ax2.set_title("Reconstruction vs. KL components")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "vae_training_curve.png", dpi=150)
    print(f"Saved training curve to {FIGURES_DIR / 'vae_training_curve.png'}")

    print_section("DONE")
    print("Training complete. Review the loss curve before building score_anomaly.py or "
          "adjusting hyperparameters (latent_dim, beta, epochs).")


if __name__ == "__main__":
    train()