#!/usr/bin/env python3
"""Cross-pipeline signal correlation for FACED differential-entropy (DE) features.

Computes Pearson r between two preprocessing pipelines (e.g., the official
Chen 2023 release and an in-house re-preprocessing) for each subject, channel,
and frequency band. Both pipelines apply different preprocessing (bandpass,
reference, normalization) to the same source recordings, so DE features should
be correlated only weakly despite representing the same neural events.

Reproduces the r approx 0.55 / r squared approx 0.30 cross-pipeline number
reported in Section 3.2 of the paper.

WHY DE-to-DE (not learned-feature-to-learned-feature):
  Learned features couple preprocessing with encoder training dynamics. The
  closed-form DE formula 0.5 log(2 pi e var) applied per band isolates
  preprocessing as the sole variable. If r is small under these maximally
  controlled conditions, preprocessing alone is sufficient to make feature
  vectors from the same recordings effectively independent.

Data layout (both directories contain sub{NNN}.pkl with dict key 'data'):
  Pipeline A: (28 trials, n_ch_a, 30 windows, 5 bands)
  Pipeline B: (28 trials, n_ch_b, 30 windows, 5 bands)
We align to the smaller channel count.

Inputs:
  --pipeline-a  Directory of pickled DE features from pipeline A (one .pkl per subject)
  --pipeline-b  Directory of pickled DE features from pipeline B (one .pkl per subject)
  --output      Output JSON path. Defaults to ./results/cross_pipeline_r_results.json
  --n-subs      Number of subjects to compare (default 123, the FACED cohort size)
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
from scipy.stats import pearsonr


BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
N_BANDS = len(BAND_NAMES)


def load_de_pkl(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        return np.asarray(obj.get("data", list(obj.values())[0]))
    return np.asarray(obj)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline-a", default="./data/faced_de_pipeline_a",
                        help="Pipeline A DE directory (default: %(default)s)")
    parser.add_argument("--pipeline-b", default="./data/faced_de_pipeline_b",
                        help="Pipeline B DE directory (default: %(default)s)")
    parser.add_argument("--output", default="./results/cross_pipeline_r_results.json",
                        help="Output JSON path (default: %(default)s)")
    parser.add_argument("--n-subs", type=int, default=123,
                        help="Number of subjects (default: %(default)s)")
    args = parser.parse_args()

    for d in [args.pipeline_a, args.pipeline_b]:
        if not os.path.isdir(d):
            print(f"ERROR: directory not found: {d}", file=sys.stderr)
            sys.exit(1)

    per_sub_per_band = np.full((args.n_subs, N_BANDS), np.nan)
    per_sub_overall = np.full(args.n_subs, np.nan)
    skipped = []

    for sub in range(args.n_subs):
        a_path = os.path.join(args.pipeline_a, f"sub{sub:03d}.pkl")
        b_path = os.path.join(args.pipeline_b, f"sub{sub:03d}.pkl")

        if not os.path.exists(a_path) or not os.path.exists(b_path):
            skipped.append(sub)
            continue

        a_de = load_de_pkl(a_path)
        b_de = load_de_pkl(b_path)

        # Align channel counts to the smaller of the two
        n_ch = min(a_de.shape[1], b_de.shape[1])
        a_de = a_de[:, :n_ch, :, :]
        b_de = b_de[:, :n_ch, :, :]

        if a_de.shape != b_de.shape:
            print(f"  sub{sub:03d}: shape mismatch a={a_de.shape} b={b_de.shape}, skipping")
            skipped.append(sub)
            continue

        for b in range(N_BANDS):
            x = a_de[:, :, :, b].flatten()
            y = b_de[:, :, :, b].flatten()
            if x.std() < 1e-12 or y.std() < 1e-12:
                continue
            r, _ = pearsonr(x, y)
            per_sub_per_band[sub, b] = r

        x_all = a_de.flatten()
        y_all = b_de.flatten()
        if x_all.std() > 1e-12 and y_all.std() > 1e-12:
            per_sub_overall[sub], _ = pearsonr(x_all, y_all)

    valid_subs = [s for s in range(args.n_subs) if s not in skipped]
    n_valid = len(valid_subs)

    print(f"\n{'='*65}")
    print(f"Cross-Pipeline DE Correlation: A vs B")
    print(f"{'='*65}")
    print(f"Subjects compared: {n_valid}/{args.n_subs} (skipped: {len(skipped)})")
    print(f"Bands: {N_BANDS}\n")

    print(f"{'Band':<10} {'Mean r':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 50)
    band_means = []
    for b in range(N_BANDS):
        vals = per_sub_per_band[valid_subs, b]
        vals = vals[~np.isnan(vals)]
        m, s = float(np.mean(vals)), float(np.std(vals))
        band_means.append(m)
        print(f"{BAND_NAMES[b]:<10} {m:>10.4f} {s:>10.4f} "
              f"{np.min(vals):>10.4f} {np.max(vals):>10.4f}")

    overall_vals = per_sub_overall[valid_subs]
    overall_vals = overall_vals[~np.isnan(overall_vals)]
    overall_mean = float(np.mean(overall_vals))
    overall_std = float(np.std(overall_vals))

    print("-" * 50)
    print(f"{'Overall':<10} {overall_mean:>10.4f} {overall_std:>10.4f} "
          f"{np.min(overall_vals):>10.4f} {np.max(overall_vals):>10.4f}")

    print(f"\nCross-pipeline r = {overall_mean:.4f} +/- {overall_std:.4f}")
    if abs(overall_mean) < 0.1:
        print("=> Near-zero correlation: the two pipelines produce effectively independent representations.")
    elif abs(overall_mean) < 0.5:
        print("=> Weak-to-moderate correlation: pipelines share some variance but are substantially different.")
    else:
        print("=> Stronger correlation than expected; verify the directories actually contain different preprocessings.")

    results = {
        "experiment": "cross_pipeline_correlation",
        "pipeline_a": args.pipeline_a,
        "pipeline_b": args.pipeline_b,
        "n_subjects": n_valid,
        "n_bands": N_BANDS,
        "overall_mean_r": overall_mean,
        "overall_std_r": overall_std,
        "per_band_mean_r": {BAND_NAMES[b]: band_means[b] for b in range(N_BANDS)},
        "per_subject_overall_r": {
            f"sub{s:03d}": float(per_sub_overall[s])
            for s in valid_subs if not np.isnan(per_sub_overall[s])
        },
        "skipped_subjects": skipped,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    main()
