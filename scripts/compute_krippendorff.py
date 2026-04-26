#!/usr/bin/env python3
"""Compute Krippendorff's alpha for FACED and DEAP self-reported labels.

Measures inter-subject agreement: do subjects rate the same stimuli similarly?

Interpretation (Krippendorff's guidelines):
  alpha > 0.80   reliable for drawing conclusions
  0.67 to 0.80   tentative conclusions only
  alpha < 0.67   unreliable, data should be discarded

For EEG emotion recognition: low alpha means subjects genuinely disagree about
what they felt, making cross-subject models fundamentally limited by label noise.

This script reproduces the Krippendorff alpha values reported in Table 3 of the
accompanying paper (Section 4.3).

Inputs:
  --faced-dir      Path to a directory containing FACED subject folders sub001..sub123
                   each with an After_remarks.mat file. Defaults to ./data/faced.
  --deap-ratings   Path to the DEAP participant_ratings.csv (the official metadata
                   file released with the DEAP corpus). Defaults to
                   ./data/deap/participant_ratings.csv.

Skips a corpus silently if its directory is missing, so the script is usable
on machines with access to only one of the two datasets.
"""
import argparse
import os
from collections import defaultdict

import numpy as np


VALENCE_IDX = 9
AROUSAL_IDX = 8

# FACED video IDs to emotion category (1-indexed)
VID_TO_EMOTION = {}
for v in range(1, 4):   VID_TO_EMOTION[v] = "anger"
for v in range(4, 7):   VID_TO_EMOTION[v] = "disgust"
for v in range(7, 10):  VID_TO_EMOTION[v] = "fear"
for v in range(10, 13): VID_TO_EMOTION[v] = "sadness"
for v in range(13, 17): VID_TO_EMOTION[v] = "neutral"
for v in range(17, 20): VID_TO_EMOTION[v] = "amusement"
for v in range(20, 23): VID_TO_EMOTION[v] = "inspiration"
for v in range(23, 26): VID_TO_EMOTION[v] = "joy"
for v in range(26, 29): VID_TO_EMOTION[v] = "tenderness"

VID_TO_VALENCE_CLASS = {}
for v, e in VID_TO_EMOTION.items():
    if e in ("anger", "disgust", "fear", "sadness"):
        VID_TO_VALENCE_CLASS[v] = "negative"
    elif e == "neutral":
        VID_TO_VALENCE_CLASS[v] = "neutral"
    else:
        VID_TO_VALENCE_CLASS[v] = "positive"


def load_faced(faced_dir):
    import scipy.io
    subjects = sorted([d for d in os.listdir(faced_dir) if d.startswith("sub")])
    n_subj = len(subjects)

    valence_matrix = np.full((n_subj, 28), np.nan)
    arousal_matrix = np.full((n_subj, 28), np.nan)
    binary_valence_matrix = np.full((n_subj, 28), np.nan)

    for s_idx, subj in enumerate(subjects):
        mat_path = os.path.join(faced_dir, subj, "After_remarks.mat")
        if not os.path.exists(mat_path):
            continue
        mat = scipy.io.loadmat(mat_path)
        ar = mat["After_remark"]

        for trial_idx in range(28):
            row = ar[trial_idx, 0]
            score = row["score"]
            if score.size == 0:
                continue
            vid = int(row["vid"][0, 0])
            valence = float(score[0, VALENCE_IDX])
            arousal = float(score[0, AROUSAL_IDX])

            vid_idx = vid - 1
            valence_matrix[s_idx, vid_idx] = valence
            arousal_matrix[s_idx, vid_idx] = arousal
            binary_valence_matrix[s_idx, vid_idx] = 1 if valence >= 3.5 else 0

    return subjects, valence_matrix, arousal_matrix, binary_valence_matrix


def load_deap(deap_ratings):
    import pandas as pd
    df = pd.read_csv(deap_ratings)

    subjects = sorted(df["Participant_id"].unique())
    n_subj = len(subjects)

    valence_matrix = np.full((n_subj, 40), np.nan)
    arousal_matrix = np.full((n_subj, 40), np.nan)
    binary_valence_matrix = np.full((n_subj, 40), np.nan)

    for s_idx, sid in enumerate(subjects):
        sub_df = df[df["Participant_id"] == sid].sort_values("Trial")
        for _, row in sub_df.iterrows():
            trial_idx = int(row["Experiment_id"]) - 1
            valence_matrix[s_idx, trial_idx] = row["Valence"]
            arousal_matrix[s_idx, trial_idx] = row["Arousal"]
            binary_valence_matrix[s_idx, trial_idx] = 1 if row["Valence"] >= 5.0 else 0

    return subjects, valence_matrix, arousal_matrix, binary_valence_matrix


def compute_alphas(name, valence_mat, arousal_mat, binary_mat, vid_to_emotion=None):
    import krippendorff

    print(f"\n{'='*70}")
    print(f"  {name}: Krippendorff's alpha")
    print(f"{'='*70}")
    print(f"  Matrix shape: {valence_mat.shape} (subjects x stimuli)")
    non_nan = np.sum(~np.isnan(valence_mat))
    total = valence_mat.size
    print(f"  Non-NaN entries: {non_nan}/{total} ({non_nan/total*100:.1f}%)")
    print()

    alpha_val = krippendorff.alpha(valence_mat, level_of_measurement="interval")
    alpha_aro = krippendorff.alpha(arousal_mat, level_of_measurement="interval")
    alpha_bin = krippendorff.alpha(binary_mat,  level_of_measurement="nominal")

    def interpret(a):
        if a > 0.80:
            return "RELIABLE"
        if a > 0.67:
            return "TENTATIVE"
        return "UNRELIABLE"

    print(f"  Overall alpha (interval, valence):   {alpha_val:.4f}  [{interpret(alpha_val)}]")
    print(f"  Overall alpha (interval, arousal):   {alpha_aro:.4f}  [{interpret(alpha_aro)}]")
    print(f"  Overall alpha (nominal, binary val): {alpha_bin:.4f}  [{interpret(alpha_bin)}]")
    print()

    if vid_to_emotion is not None:
        print(f"  Per-emotion-category alpha (interval valence):")
        print(f"  {'Category':>15}  {'alpha':>8}  {'Interp':>12}  {'Videos':>8}")
        print(f"  {'-'*15}  {'-'*8}  {'-'*12}  {'-'*8}")

        categories = ["anger", "disgust", "fear", "sadness", "neutral",
                      "amusement", "inspiration", "joy", "tenderness"]
        for cat in categories:
            vid_indices = [v - 1 for v, e in vid_to_emotion.items() if e == cat]
            if len(vid_indices) < 2:
                continue
            subset = valence_mat[:, vid_indices]
            try:
                a = krippendorff.alpha(subset, level_of_measurement="interval")
                print(f"  {cat:>15}  {a:>8.4f}  {interpret(a):>12}  {len(vid_indices):>8}")
            except Exception as e:
                print(f"  {cat:>15}  ERROR: {e}")
        print()

        neg_vids = [v - 1 for v, c in VID_TO_VALENCE_CLASS.items() if c == "negative"]
        pos_vids = [v - 1 for v, c in VID_TO_VALENCE_CLASS.items() if c == "positive"]
        neu_vids = [v - 1 for v, c in VID_TO_VALENCE_CLASS.items() if c == "neutral"]

        for group, indices in [("Negative (12 vids)", neg_vids),
                               ("Positive (12 vids)", pos_vids),
                               ("Neutral (4 vids)",   neu_vids)]:
            subset = valence_mat[:, indices]
            a = krippendorff.alpha(subset, level_of_measurement="interval")
            print(f"  {group:>20}  alpha = {a:.4f}  [{interpret(a)}]")

        print()
        print(f"  Stimulus to self-report binary agreement (alpha, nominal):")
        stim_binary = np.zeros((1, valence_mat.shape[1]))
        for vid_idx in range(28):
            vid = vid_idx + 1
            cls = VID_TO_VALENCE_CLASS.get(vid)
            if cls == "positive":
                stim_binary[0, vid_idx] = 1
            elif cls == "negative":
                stim_binary[0, vid_idx] = 0
            else:
                stim_binary[0, vid_idx] = np.nan

        per_subj_agreement = []
        for s in range(valence_mat.shape[0]):
            non_neutral = ~np.isnan(stim_binary[0]) & ~np.isnan(binary_mat[s])
            if non_neutral.sum() > 0:
                agree = (stim_binary[0, non_neutral] == binary_mat[s, non_neutral]).mean()
                per_subj_agreement.append(agree)

        agreements = np.array(per_subj_agreement)
        print(f"    Mean per-subject agreement: {agreements.mean()*100:.1f}% +/- {agreements.std()*100:.1f}%")
        print(f"    Min: {agreements.min()*100:.1f}%, Max: {agreements.max()*100:.1f}%")
        print(f"    (100% - mean = {(1-agreements.mean())*100:.1f}% average mismatch rate)")

    return alpha_val, alpha_aro, alpha_bin


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--faced-dir",    default="./data/faced",
                        help="FACED root with sub*/After_remarks.mat (default: %(default)s)")
    parser.add_argument("--deap-ratings", default="./data/deap/participant_ratings.csv",
                        help="DEAP participant_ratings.csv path (default: %(default)s)")
    args = parser.parse_args()

    results = {}
    if os.path.isdir(args.faced_dir):
        print(f"Loading FACED self-report labels from {args.faced_dir} ...")
        _, faced_val, faced_aro, faced_bin = load_faced(args.faced_dir)
        results["FACED"] = compute_alphas(
            "FACED (123 subjects x 28 videos)",
            faced_val, faced_aro, faced_bin,
            vid_to_emotion=VID_TO_EMOTION,
        )
    else:
        print(f"[skip] FACED dir not found: {args.faced_dir}")

    if os.path.isfile(args.deap_ratings):
        print(f"\nLoading DEAP self-report labels from {args.deap_ratings} ...")
        _, deap_val, deap_aro, deap_bin = load_deap(args.deap_ratings)
        results["DEAP"] = compute_alphas(
            "DEAP (32 subjects x 40 videos)",
            deap_val, deap_aro, deap_bin,
        )
    else:
        print(f"[skip] DEAP ratings not found: {args.deap_ratings}")

    if results:
        print(f"\n{'='*70}")
        print(f"  SUMMARY: Cross-Dataset Label Reliability")
        print(f"{'='*70}")
        print(f"  {'':>20}  {'Valence(int)':>14}  {'Arousal(int)':>14}  {'Binary(nom)':>14}")
        for name, (av, aa, ab) in results.items():
            print(f"  {name:>20}  {av:>14.4f}  {aa:>14.4f}  {ab:>14.4f}")
        print()
        print("  alpha < 0.67 = UNRELIABLE; 0.67-0.80 = TENTATIVE; alpha > 0.80 = RELIABLE.")
        print("  Low alpha bounds the cross-subject classifier accuracy ceiling on stimulus-driven features.")


if __name__ == "__main__":
    main()
