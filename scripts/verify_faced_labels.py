#!/usr/bin/env python3
"""Verify FACED valence label statistics cited in the paper.

Claims verified:
  1. The FACED ambiguous zone (analogous to the DEAP 4.0-6.0 zone, scaled to 0-7)
  2. 19.8% stimulus vs. self-report binary mismatch (~585/2952 non-neutral trials)
  3. Per-subject mean valence span across 123 subjects
  4. Per-category mismatch: negatives 10-16%, positives 23-31%

Inputs:
  --faced-dir   FACED root with sub*/After_remarks.mat (default: ./data/faced)
"""
import argparse
import os

import numpy as np


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

STIMULUS_VALENCE = {
    "anger": "negative", "disgust": "negative", "fear": "negative", "sadness": "negative",
    "neutral": "neutral",
    "amusement": "positive", "inspiration": "positive", "joy": "positive", "tenderness": "positive",
}

VALENCE_IDX = 9
AROUSAL_IDX = 8


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--faced-dir", default="./data/faced",
                        help="FACED root with sub*/After_remarks.mat (default: %(default)s)")
    args = parser.parse_args()
    import scipy.io

    subjects = sorted([d for d in os.listdir(args.faced_dir) if d.startswith("sub")])
    print(f"Found {len(subjects)} subjects")

    all_valence = []
    per_subject_mean_valence = {}
    per_trial_data = []

    for subj in subjects:
        mat_path = os.path.join(args.faced_dir, subj, "After_remarks.mat")
        if not os.path.exists(mat_path):
            print(f"  WARNING: missing {mat_path}")
            continue

        mat = scipy.io.loadmat(mat_path)
        ar = mat["After_remark"]

        subj_valences = []
        for trial_idx in range(28):
            row = ar[trial_idx, 0]
            score = row["score"]
            if score.size == 0:
                continue
            valence = float(score[0, VALENCE_IDX])
            arousal = float(score[0, AROUSAL_IDX])
            vid = int(row["vid"][0, 0])
            emotion = VID_TO_EMOTION[vid]
            stim_valence = STIMULUS_VALENCE[emotion]
            sr_valence = "positive" if valence >= 3.5 else "negative"

            per_trial_data.append({
                "subject": subj, "trial": trial_idx, "emotion": emotion,
                "stim_valence": stim_valence, "sr_valence_raw": valence,
                "sr_arousal_raw": arousal, "sr_valence_binary": sr_valence,
            })
            subj_valences.append(valence)
            all_valence.append(valence)

        if subj_valences:
            per_subject_mean_valence[subj] = float(np.mean(subj_valences))

    all_valence = np.array(all_valence)
    print(f"Total trials with valence data: {len(all_valence)}")
    print(f"Valence range: {all_valence.min():.2f} - {all_valence.max():.2f}")
    print(f"Valence mean: {np.mean(all_valence):.2f}, std: {np.std(all_valence):.2f}")
    print()

    print("=" * 60)
    print("CLAIM 1: FACED ambiguous zone")
    zones = [
        ("Proportional to DEAP [4,6] on 0-7 scale: [3.11, 4.67]", 3.11, 4.67),
        ("+/-1.0 of midpoint [2.5, 4.5]", 2.5, 4.5),
        ("+/-1.5 of midpoint [2.0, 5.0]", 2.0, 5.0),
    ]
    for desc, lo, hi in zones:
        n = int(((all_valence >= lo) & (all_valence <= hi)).sum())
        pct = n / len(all_valence) * 100
        print(f"  {desc}: {n}/{len(all_valence)} = {pct:.1f}%")
    print()

    print("=" * 60)
    print("CLAIM 2: Stimulus vs self-report valence mismatch")
    non_neutral = [d for d in per_trial_data if d["stim_valence"] != "neutral"]
    total_non_neutral = len(non_neutral)
    mismatches = [d for d in non_neutral if d["stim_valence"] != d["sr_valence_binary"]]
    pct_mismatch = len(mismatches) / total_non_neutral * 100
    print(f"  Non-neutral trials: {total_non_neutral}")
    print(f"  Mismatches: {len(mismatches)}/{total_non_neutral} = {pct_mismatch:.1f}%")
    print(f"  Paper claims: ~19.8%")
    print(f"  MATCH: {'YES' if abs(pct_mismatch - 19.8) < 1 else 'NO -- INVESTIGATE'}")
    print()

    print("  Per-category mismatch rates:")
    for cat in ["anger", "disgust", "fear", "sadness", "amusement", "inspiration", "joy", "tenderness"]:
        cat_trials = [d for d in non_neutral if d["emotion"] == cat]
        cat_mismatches = [d for d in cat_trials if d["stim_valence"] != d["sr_valence_binary"]]
        if cat_trials:
            consistency = 1 - len(cat_mismatches) / len(cat_trials)
            print(f"    {cat:>12}: {len(cat_mismatches):>3}/{len(cat_trials):>3} mismatch "
                  f"= {len(cat_mismatches)/len(cat_trials)*100:.1f}% (consistency: {consistency*100:.1f}%)")
    print()

    print("=" * 60)
    print("CLAIM 3: Per-subject mean valence span")
    means = sorted(per_subject_mean_valence.items(), key=lambda x: x[1])
    print(f"  Lowest mean: {means[0][0]} = {means[0][1]:.2f}")
    print(f"  Highest mean: {means[-1][0]} = {means[-1][1]:.2f}")
    span = means[-1][1] - means[0][1]
    print(f"  Span: {span:.2f} points")

    print()
    print("=" * 60)
    print("EXTRA: FACED self-report majority baselines")
    for t in [3.0, 3.5, 4.0]:
        high = int((all_valence >= t).sum())
        low = int((all_valence < t).sum())
        majority = max(high, low) / len(all_valence) * 100
        print(f"  Threshold {t}: high={high}, low={low}, majority={majority:.1f}%")


if __name__ == "__main__":
    main()
