#!/usr/bin/env python3
"""Show that DEAP preprocessed data has no signal below 4 Hz (phantom delta band).

Computes PSD from DEAP .dat files and plots the power spectrum, demonstrating
that the 4-45 Hz bandpass filter removes all delta-band content. Papers extracting
delta (1-4 Hz) features from this data are extracting noise/roll-off artifacts.
"""
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch

DATA_DIR = os.environ.get('DEAP_DIR', './data/deap')
OUT_DIR = os.environ.get('OUTPUT_DIR', './output/figures')
os.makedirs(OUT_DIR, exist_ok=True)

FS = 128  # DEAP sampling rate

subjects_to_plot = [1, 10, 20, 32]
band_defs = {
    'delta (1-4 Hz)': (1, 4),
    'theta (4-8 Hz)': (4, 8),
    'alpha (8-14 Hz)': (8, 14),
    'beta (14-31 Hz)': (14, 31),
    'gamma (31-45 Hz)': (31, 45),
}

fig, axes = plt.subplots(2, 2, figsize=(8, 5), sharex=True, sharey=True)
axes = axes.flatten()

band_powers_all = {b: [] for b in band_defs}

for idx, subj in enumerate(subjects_to_plot):
    fname = os.path.join(DATA_DIR, f's{subj:02d}.dat')
    with open(fname, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    eeg = data['data'][:, :32, :]  # (40 trials, 32 channels, 8064 samples)
    eeg_concat = eeg.reshape(-1, eeg.shape[-1])  # (40*32, 8064)

    # Average PSD across all trials and channels
    freqs, psd = welch(eeg_concat, fs=FS, nperseg=256, noverlap=128)
    psd_mean = psd.mean(axis=0)

    ax = axes[idx]
    ax.semilogy(freqs, psd_mean, 'k-', linewidth=0.8)

    # Shade frequency bands
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    for (bname, (f_lo, f_hi)), color in zip(band_defs.items(), colors):
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        ax.fill_between(freqs[mask], psd_mean[mask], alpha=0.3, color=color, label=bname)
        band_powers_all[bname].append(psd_mean[mask].mean())

    ax.axvline(x=4, color='red', linestyle='--', linewidth=1.0, alpha=0.7)
    ax.set_title(f'Subject {subj}', fontsize=9)
    ax.set_xlim(0, 50)

    if idx >= 2:
        ax.set_xlabel('Frequency (Hz)', fontsize=9)
    if idx % 2 == 0:
        ax.set_ylabel('PSD (µV²/Hz)', fontsize=9)

axes[0].legend(fontsize=6, loc='upper right', ncol=1)

fig.suptitle('DEAP preprocessed data PSD — 4 Hz high-pass clearly visible', fontsize=10, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUT_DIR, 'phantom_delta_psd.pdf'), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(OUT_DIR, 'phantom_delta_psd.png'), bbox_inches='tight', dpi=300)

# Print band power comparison
print("\n=== Band Power Analysis (mean PSD across subjects) ===\n")
print(f"{'Band':<25} {'Mean PSD':>12} {'Ratio to theta':>16}")
print("-" * 55)
theta_power = np.mean(band_powers_all['theta (4-8 Hz)'])
for bname, powers in band_powers_all.items():
    mean_p = np.mean(powers)
    ratio = mean_p / theta_power
    marker = " *** PHANTOM" if 'delta' in bname else ""
    print(f"{bname:<25} {mean_p:>12.4f} {ratio:>16.4f}{marker}")

print(f"\nDelta/theta power ratio: {np.mean(band_powers_all['delta (1-4 Hz)'])/theta_power:.4f}")
print("A ratio << 1.0 confirms delta band contains only filter roll-off, not neural signal.")
print(f"\nSaved to {OUT_DIR}/phantom_delta_psd.pdf")
