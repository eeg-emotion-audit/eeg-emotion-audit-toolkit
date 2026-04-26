#!/usr/bin/env python3
"""Compute the noise transition matrix P(self-report | stimulus) for FACED.

Treats stimulus-based labels as 'intended' and self-report as 'experienced',
quantifying the gap between the two label paradigms. Reports the conditional
entropy H(SR | stim), the mutual information I(stim; SR), and per-emotion
agreement rates.

Reproduces the 19.8% binary mismatch + 29.4% MI reduction figures cited in
Section 4.4 of the paper.

Inputs:
  --faced-dir   FACED root with sub*/After_remarks.mat (default: ./data/faced)
"""
import argparse
import os

import numpy as np


VALENCE_IDX = 9

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

EMOTION_TO_VALENCE = {
    "anger": "negative", "disgust": "negative", "fear": "negative", "sadness": "negative",
    "neutral": "neutral",
    "amusement": "positive", "inspiration": "positive", "joy": "positive", "tenderness": "positive",
}


def load_all_trials(faced_dir):
    import scipy.io
    subjects = sorted([d for d in os.listdir(faced_dir) if d.startswith("sub")])
    trials = []
    for subj in subjects:
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
            emotion = VID_TO_EMOTION[vid]
            stim_valence = EMOTION_TO_VALENCE[emotion]
            sr_binary = "positive" if valence >= 3.5 else "negative"
            trials.append({
                "subject": subj,
                "vid": vid,
                "emotion": emotion,
                "stim_valence": stim_valence,
                "sr_valence": valence,
                "sr_binary": sr_binary,
            })
    return trials


def entropy(probs):
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--faced-dir", default="./data/faced",
                        help="FACED root with sub*/After_remarks.mat (default: %(default)s)")
    args = parser.parse_args()

    trials = load_all_trials(args.faced_dir)
    print(f"Loaded {len(trials)} trials")

    print("\n" + "=" * 70)
    print("  NOISE TRANSITION MATRIX: P(self-report | stimulus)")
    print("  Rows = stimulus (intended), Columns = self-report (experienced)")
    print("=" * 70)

    non_neutral = [t for t in trials if t["stim_valence"] != "neutral"]
    labels = ["negative", "positive"]
    matrix = np.zeros((2, 2))
    for t in non_neutral:
        i = labels.index(t["stim_valence"])
        j = labels.index(t["sr_binary"])
        matrix[i, j] += 1

    row_sums = matrix.sum(axis=1, keepdims=True)
    prob_matrix = matrix / row_sums

    print(f"\n  Binary (N={len(non_neutral)}, excludes neutral stimuli):")
    print(f"  {'':>15} | {'SR=negative':>12} {'SR=positive':>12} | {'Total':>8}")
    print(f"  {'-'*15}-+-{'-'*12}-{'-'*12}-+-{'-'*8}")
    for i, label in enumerate(labels):
        print(f"  {'Stim='+label:>15} | {prob_matrix[i,0]:>11.1%} {prob_matrix[i,1]:>11.1%} | {int(row_sums[i,0]):>8}")

    diag_mean = float(np.diag(prob_matrix).mean())
    off_diag_mean = float((prob_matrix.sum() - np.trace(prob_matrix)) / (prob_matrix.size - len(labels)))
    print(f"\n  Diagonal mean (agreement): {diag_mean:.1%}")
    print(f"  Off-diagonal mean (mismatch): {off_diag_mean:.1%}")

    print(f"\n{'='*70}")
    print("  PER-CATEGORY: P(self-report binary | emotion category)")
    print("=" * 70)

    categories = ["anger", "disgust", "fear", "sadness", "neutral",
                  "amusement", "inspiration", "joy", "tenderness"]
    print(f"\n  {'Category':>15} {'Stim val':>10} {'SR=neg':>8} {'SR=pos':>8} {'P(match)':>10} {'N':>6}")
    print(f"  {'-'*15} {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*6}")

    for cat in categories:
        cat_trials = [t for t in trials if t["emotion"] == cat]
        n = len(cat_trials)
        n_neg = sum(1 for t in cat_trials if t["sr_binary"] == "negative")
        n_pos = n - n_neg
        expected = EMOTION_TO_VALENCE[cat]
        if expected == "negative":
            match_rate = n_neg / n
        elif expected == "positive":
            match_rate = n_pos / n
        else:
            match_rate = float("nan")
        print(f"  {cat:>15} {expected:>10} {n_neg:>8} {n_pos:>8} {match_rate:>9.1%} {n:>6}")

    print(f"\n{'='*70}")
    print("  CONDITIONAL ENTROPY: H(self-report | stimulus)")
    print("=" * 70)

    h_sr_given_stim = 0.0
    for i in range(len(labels)):
        p_stim = row_sums[i, 0] / len(non_neutral)
        h_sr_given_stim += p_stim * entropy(prob_matrix[i])
    h_sr = entropy(np.array([
        sum(1 for t in non_neutral if t["sr_binary"] == l) / len(non_neutral)
        for l in labels
    ]))

    print(f"\n  H(self-report) = {h_sr:.4f} bits")
    print(f"  H(self-report | stimulus) = {h_sr_given_stim:.4f} bits")
    print(f"  I(stimulus; self-report) = {h_sr - h_sr_given_stim:.4f} bits")
    print(f"  Normalized MI = {(h_sr - h_sr_given_stim) / h_sr:.4f}")
    print(f"  Stimulus reduces uncertainty about self-report by {(h_sr - h_sr_given_stim)/h_sr*100:.1f}%.")
    print(f"  The remaining {h_sr_given_stim/h_sr*100:.1f}% is individual variation.")

    print(f"\n{'='*70}")
    print("  ASYMMETRY: negative vs positive stimulus mismatch rates")
    print("=" * 70)

    neg_stim = [t for t in non_neutral if t["stim_valence"] == "negative"]
    pos_stim = [t for t in non_neutral if t["stim_valence"] == "positive"]
    neg_flip = sum(1 for t in neg_stim if t["sr_binary"] == "positive") / len(neg_stim)
    pos_flip = sum(1 for t in pos_stim if t["sr_binary"] == "negative") / len(pos_stim)
    print(f"\n  Negative stimulus -> self-report positive: {neg_flip:.1%}")
    print(f"  Positive stimulus -> self-report negative: {pos_flip:.1%}")
    if neg_flip > 0:
        print(f"  Asymmetry: positive stimuli are {pos_flip/neg_flip:.1f}x noisier than negative.")


if __name__ == "__main__":
    main()
