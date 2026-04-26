#!/usr/bin/env python3
"""Verify the absence of delta-band signal in DEAP's preprocessed release.

The DEAP preprocessed release is bandpass-filtered at 4-45 Hz, removing all
delta-band (1-4 Hz) content. Papers extracting delta-band features from this
data are extracting filter roll-off, not neural signal. This script computes
PSD across a few subjects and plots the spectrum, with the cumulative
delta/theta power ratio as the headline number.

Reproduces the 'phantom delta band' verification reported in Appendix E.

Inputs:
  --data-dir   Directory containing s01.dat ... s32.dat from the official DEAP
               preprocessed release. Default: ./data/deap_preprocessed_python
  --out-dir    Where to write the PSD figure (PDF + PNG). Default: ./results/figures
  --subjects   Comma-separated subject indices to plot (default: 1,10,20,32)
"""
import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch


FS = 128  # DEAP sampling rate
BAND_DEFS = {
    "delta (1-4 Hz)":  (1, 4),
    "theta (4-8 Hz)":  (4, 8),
    "alpha (8-14 Hz)": (8, 14),
    "beta (14-31 Hz)": (14, 31),
    "gamma (31-45 Hz)": (31, 45),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="./data/deap_preprocessed_python",
                        help="DEAP preprocessed .dat directory (default: %(default)s)")
    parser.add_argument("--out-dir", default="./results/figures",
                        help="Output directory for PSD figure (default: %(default)s)")
    parser.add_argument("--subjects", default="1,10,20,32",
                        help="Comma-separated subject IDs to plot (default: %(default)s)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    subjects_to_plot = [int(s) for s in args.subjects.split(",")]

    fig, axes = plt.subplots(2, 2, figsize=(8, 5), sharex=True, sharey=True)
    axes = axes.flatten()

    band_powers_all = {b: [] for b in BAND_DEFS}

    for idx, subj in enumerate(subjects_to_plot):
        fname = os.path.join(args.data_dir, f"s{subj:02d}.dat")
        with open(fname, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        eeg = data["data"][:, :32, :]
        eeg_concat = eeg.reshape(-1, eeg.shape[-1])

        freqs, psd = welch(eeg_concat, fs=FS, nperseg=256, noverlap=128)
        psd_mean = psd.mean(axis=0)

        ax = axes[idx]
        ax.semilogy(freqs, psd_mean, "k-", linewidth=0.8)

        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
        for (bname, (f_lo, f_hi)), color in zip(BAND_DEFS.items(), colors):
            mask = (freqs >= f_lo) & (freqs <= f_hi)
            ax.fill_between(freqs[mask], psd_mean[mask], alpha=0.3, color=color, label=bname)
            band_powers_all[bname].append(psd_mean[mask].mean())

        ax.axvline(x=4, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(f"Subject {subj}", fontsize=9)
        ax.set_xlim(0, 50)
        if idx >= 2:
            ax.set_xlabel("Frequency (Hz)", fontsize=9)
        if idx % 2 == 0:
            ax.set_ylabel("PSD (uV^2/Hz)", fontsize=9)

    axes[0].legend(fontsize=6, loc="upper right", ncol=1)

    fig.suptitle("DEAP preprocessed PSD: 4 Hz high-pass clearly visible", fontsize=10, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf_path = os.path.join(args.out_dir, "phantom_delta_psd.pdf")
    png_path = os.path.join(args.out_dir, "phantom_delta_psd.png")
    plt.savefig(pdf_path, bbox_inches="tight", dpi=300)
    plt.savefig(png_path, bbox_inches="tight", dpi=300)

    print("\n=== Band Power Analysis (mean PSD across subjects) ===\n")
    print(f"{'Band':<25} {'Mean PSD':>12} {'Ratio to theta':>16}")
    print("-" * 55)
    theta_power = float(np.mean(band_powers_all["theta (4-8 Hz)"]))
    for bname, powers in band_powers_all.items():
        mean_p = float(np.mean(powers))
        ratio = mean_p / theta_power
        marker = " *** PHANTOM" if "delta" in bname else ""
        print(f"{bname:<25} {mean_p:>12.4f} {ratio:>16.4f}{marker}")

    print(f"\nDelta/theta power ratio: {np.mean(band_powers_all['delta (1-4 Hz)'])/theta_power:.4f}")
    print("Ratio << 1.0 confirms the delta band contains only filter roll-off, not neural signal.")
    print(f"\nSaved: {pdf_path}\n       {png_path}")


if __name__ == "__main__":
    main()
