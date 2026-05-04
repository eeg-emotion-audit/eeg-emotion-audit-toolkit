#!/usr/bin/env python3
"""S3 — Cross-pipeline signal correlation for FACED DE features.

Computes Pearson r between Chen 2023 and DAEST-preprocessed DE features
for each subject, channel, and frequency band. The two pipelines apply
completely different preprocessing (bandpass, reference, normalization)
to the same source recordings, so DE features should be near-uncorrelated
despite representing the same neural events.

This backs up the abstract claim "cross-pipeline signal correlation r ≈ 0"
with an actual computed number.

WHY DE-to-DE (not ME-to-ME):
  DAEST's published results (67.9/81.7%) use ME (marginal entropy) features,
  which are mean activations from a learned encoder — not a closed-form
  formula. Chen 2023's published results use DE (differential entropy),
  which is 0.5*log(2*pi*e*var) applied per frequency band. Computing
  "Chen ME" would require training DAEST's encoder on Chen-preprocessed
  data, confounding preprocessing with encoder training dynamics.

  Instead, we apply the SAME closed-form DE formula to BOTH preprocessed
  signals. This isolates preprocessing as the sole variable: same subjects,
  same electrodes, same frequency bands, same feature extraction code,
  different input signal. If r ≈ 0 under these maximally-controlled
  conditions, preprocessing alone is sufficient to make feature vectors
  from the same recordings effectively independent — and any downstream
  comparison across pipelines (regardless of DE or ME) is confounded by
  this preprocessing divergence.

  The DAEST-side DE features were extracted by our extract_de_from_daest_
  official.py script, which applies Chen 2023's exact DE formula (bandpass
  + time-domain variance, verified against Chen's features_extract.py) to
  DAEST's data_all_cleaned signal. This ensures the only difference is the
  preprocessing applied to the raw recordings before DE extraction.

Data layout (both dirs contain sub{:03d}.pkl with dict key 'data'):
  Chen 2023:  (28, 32, 30, 5) = (trials, channels, 1s-windows, 5 bands)
  DAEST:      (28, 30, 30, 5) = same, but 30 channels (mastoids dropped)

We align to 30 channels (first 30 of Chen = same electrodes as DAEST).

Run on triton:
    cd .
    conda activate torcheeg_env
    python semi_supervised_learning/neurips_verification/compute_cross_pipeline_r.py

CPU-only, ~2 min for 123 subjects.
"""

import json
import os
import pickle
import sys

import numpy as np
from scipy.stats import pearsonr

CHEN_DE_DIR = os.environ.get('CHEN_DE_DIR', './data/faced/de_features')
DAEST_DE_DIR = os.environ.get('DAEST_DE_DIR', './data/faced/de_features_daest')
OUT_DIR = os.path.expanduser(
    "./scripts"
)

N_SUBS = 123
N_CHANNELS = 30
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
N_BANDS = len(BAND_NAMES)


def load_de_pkl(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        return np.asarray(obj.get("data", list(obj.values())[0]))
    return np.asarray(obj)


def main():
    for d in [CHEN_DE_DIR, DAEST_DE_DIR]:
        if not os.path.isdir(d):
            print(f"ERROR: directory not found: {d}")
            print("This script must run on triton where both DE directories exist.")
            sys.exit(1)

    per_sub_per_band = np.full((N_SUBS, N_BANDS), np.nan)
    per_sub_overall = np.full(N_SUBS, np.nan)
    skipped = []

    for sub in range(N_SUBS):
        chen_path = os.path.join(CHEN_DE_DIR, f"sub{sub:03d}.pkl")
        daest_path = os.path.join(DAEST_DE_DIR, f"sub{sub:03d}.pkl")

        if not os.path.exists(chen_path):
            chen_path = chen_path + ".pkl"
        if not os.path.exists(chen_path) or not os.path.exists(daest_path):
            skipped.append(sub)
            continue

        chen_de = load_de_pkl(chen_path)
        daest_de = load_de_pkl(daest_path)

        chen_de = chen_de[:, :N_CHANNELS, :, :]

        if chen_de.shape != daest_de.shape:
            print(f"  sub{sub:03d}: shape mismatch chen={chen_de.shape} "
                  f"daest={daest_de.shape}, skipping")
            skipped.append(sub)
            continue

        for b in range(N_BANDS):
            a = chen_de[:, :, :, b].flatten()
            d = daest_de[:, :, :, b].flatten()
            if a.std() < 1e-12 or d.std() < 1e-12:
                continue
            r, _ = pearsonr(a, d)
            per_sub_per_band[sub, b] = r

        a_all = chen_de.flatten()
        d_all = daest_de.flatten()
        if a_all.std() > 1e-12 and d_all.std() > 1e-12:
            per_sub_overall[sub], _ = pearsonr(a_all, d_all)

    valid_subs = [s for s in range(N_SUBS) if s not in skipped]
    n_valid = len(valid_subs)

    print(f"\n{'='*65}")
    print(f"Cross-Pipeline DE Correlation: Chen 2023 vs DAEST (FACED)")
    print(f"{'='*65}")
    print(f"Subjects: {n_valid}/{N_SUBS} (skipped: {skipped if skipped else 'none'})")
    print(f"Channels: {N_CHANNELS}, Bands: {N_BANDS}, "
          f"Vectors per subject: 28 trials × 30 windows = 840 per (ch, band)\n")

    print(f"{'Band':<10} {'Mean r':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 50)
    band_means = []
    for b in range(N_BANDS):
        vals = per_sub_per_band[valid_subs, b]
        vals = vals[~np.isnan(vals)]
        m, s = np.mean(vals), np.std(vals)
        band_means.append(m)
        print(f"{BAND_NAMES[b]:<10} {m:>10.4f} {s:>10.4f} "
              f"{np.min(vals):>10.4f} {np.max(vals):>10.4f}")

    overall_vals = per_sub_overall[valid_subs]
    overall_vals = overall_vals[~np.isnan(overall_vals)]
    overall_mean = np.mean(overall_vals)
    overall_std = np.std(overall_vals)

    print("-" * 50)
    print(f"{'Overall':<10} {overall_mean:>10.4f} {overall_std:>10.4f} "
          f"{np.min(overall_vals):>10.4f} {np.max(overall_vals):>10.4f}")

    print(f"\nCross-pipeline r = {overall_mean:.4f} ± {overall_std:.4f}")
    if abs(overall_mean) < 0.1:
        print("=> Near-zero correlation confirms the two pipelines produce "
              "effectively independent feature representations.")
    elif abs(overall_mean) < 0.3:
        print("=> Weak correlation — pipelines share some variance but are "
              "substantially different.")
    else:
        print(f"=> r = {overall_mean:.2f} is higher than expected. "
              "Investigate whether both directories actually contain "
              "different preprocessings.")

    results = {
        "experiment": "S3_cross_pipeline_correlation",
        "chen_de_dir": CHEN_DE_DIR,
        "daest_de_dir": DAEST_DE_DIR,
        "n_subjects": n_valid,
        "n_channels": N_CHANNELS,
        "n_bands": N_BANDS,
        "overall_mean_r": float(overall_mean),
        "overall_std_r": float(overall_std),
        "per_band_mean_r": {
            BAND_NAMES[b]: float(band_means[b]) for b in range(N_BANDS)
        },
        "per_subject_overall_r": {
            f"sub{s:03d}": float(per_sub_overall[s])
            for s in valid_subs if not np.isnan(per_sub_overall[s])
        },
        "skipped_subjects": skipped,
    }

    out_path = os.path.join(OUT_DIR, "cross_pipeline_r_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
