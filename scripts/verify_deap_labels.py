#!/usr/bin/env python3
"""Verify DEAP label statistics cited in the paper.

Claims verified:
  1. ~30% of trials fall in the ambiguous valence zone (4.0-6.0 on 1-9 SAM)
  2. Per-subject calibration spans roughly 4.12 to 6.16 mean valence
  3. Seven binarization thresholds shift the majority-class baseline from
     56.5% to 72.2% (Table 2 / Section 4.1)

Inputs:
  --deap-ratings   Path to participant_ratings.csv (default: ./data/deap/participant_ratings.csv)
"""
import argparse

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deap-ratings", default="./data/deap/participant_ratings.csv",
                        help="DEAP participant_ratings.csv path (default: %(default)s)")
    args = parser.parse_args()

    df = pd.read_csv(args.deap_ratings)
    total = len(df)
    print(f"Loaded {total} trials from {df['Participant_id'].nunique()} subjects")
    print(f"Valence range: {df['Valence'].min():.2f} - {df['Valence'].max():.2f}")
    print(f"Arousal range: {df['Arousal'].min():.2f} - {df['Arousal'].max():.2f}")
    print()

    print("=" * 60)
    print("CLAIM 1: Ambiguous zone (valence 4.0-6.0)")
    ambiguous = df[(df["Valence"] >= 4.0) & (df["Valence"] <= 6.0)]
    pct = len(ambiguous) / total * 100
    print(f"  Result: {len(ambiguous)}/{total} = {pct:.1f}%")
    print(f"  Paper claims: ~30.1%")
    print(f"  MATCH: {'YES' if abs(pct - 30.1) < 1 else 'NO -- INVESTIGATE'}")
    near_threshold = df[(df["Valence"] >= 4.75) & (df["Valence"] <= 5.25)]
    print(f"  Near threshold (4.75-5.25): {len(near_threshold)}/{total} = {len(near_threshold)/total*100:.1f}%")
    print()

    print("=" * 60)
    print("CLAIM 2: Per-subject mean valence")
    per_subj = df.groupby("Participant_id")["Valence"].mean().sort_values()
    print(f"  Lowest: s{per_subj.index[0]:02d} = {per_subj.iloc[0]:.2f}")
    print(f"  Highest: s{per_subj.index[-1]:02d} = {per_subj.iloc[-1]:.2f}")
    print(f"  Range: {per_subj.max() - per_subj.min():.2f} points")
    for sid in [4, 6]:
        if sid in per_subj.index:
            print(f"  s{sid:02d} mean valence: {per_subj[sid]:.2f}")
    print(f"  Paper claims: s04 ~ 4.12, s06 ~ 6.16")
    print()

    print("=" * 60)
    print("CLAIM 3: Majority-class baseline across thresholds")
    print()
    print(f"  {'Threshold':>12}  {'High':>6}  {'Low':>6}  {'Majority%':>10}")
    print(f"  {'-'*12}  {'-'*6}  {'-'*6}  {'-'*10}")
    for t in [4.0, 4.5, 5.0, 5.25, 5.5, 6.0]:
        high = int((df["Valence"] >= t).sum())
        low = int((df["Valence"] < t).sum())
        majority = max(high, low) / total * 100
        print(f"  {'>='+str(t):>12}  {high:>6}  {low:>6}  {majority:>9.1f}%")
    print()
    print("  Alternative: threshold <=X means low")
    for t in [4, 5]:
        low = int((df["Valence"] <= t).sum())
        high = int((df["Valence"] > t).sum())
        majority = max(high, low) / total * 100
        print(f"  {'<='+str(t):>12}  {high:>6}  {low:>6}  {majority:>9.1f}%")
    print()
    print("  Paper claims: 56.5% (threshold <= 5) to 72.2% (threshold <= 4)")
    print()

    print("=" * 60)
    print("EXTRA: Valence histogram (1-point bins)")
    for low_edge in range(1, 10):
        high_edge = low_edge + 1
        count = int(((df["Valence"] >= low_edge) & (df["Valence"] < high_edge)).sum())
        bar = "#" * (count // 5)
        print(f"  [{low_edge}-{high_edge}): {count:>4}  {bar}")

    print()
    print("=" * 60)
    print("EXTRA: Arousal ambiguous zone (4.0-6.0)")
    arousal_ambig = df[(df["Arousal"] >= 4.0) & (df["Arousal"] <= 6.0)]
    print(f"  {len(arousal_ambig)}/{total} = {len(arousal_ambig)/total*100:.1f}%")
    per_subj_arousal = df.groupby("Participant_id")["Arousal"].mean()
    print(f"  Per-subject arousal range: {per_subj_arousal.min():.2f} - {per_subj_arousal.max():.2f} "
          f"({per_subj_arousal.max()-per_subj_arousal.min():.2f} span)")


if __name__ == "__main__":
    main()
