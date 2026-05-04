"""
Verify DEAP label statistics for NeurIPS paper.
Claims to verify:
  1. 30% of trials (385/1280) fall in ambiguous zone (4.0-6.0)
  2. Per-subject calibration: s04 mean=4.12, s06 mean=6.16
  3. 7 binarization thresholds → majority baseline 56.5%-72.2%
"""
import os
import pandas as pd
import numpy as np
import sys

DEAP_RATINGS = os.environ.get('DEAP_RATINGS', './data/deap/participant_ratings.csv')

df = pd.read_csv(DEAP_RATINGS)
print(f"Loaded {len(df)} trials from {df['Participant_id'].nunique()} subjects")
print(f"Valence range: {df['Valence'].min():.2f} - {df['Valence'].max():.2f}")
print(f"Arousal range: {df['Arousal'].min():.2f} - {df['Arousal'].max():.2f}")
print()

# ─── Claim 1: Ambiguous zone ───
total = len(df)
ambiguous = df[(df['Valence'] >= 4.0) & (df['Valence'] <= 6.0)]
n_ambig = len(ambiguous)
pct = n_ambig / total * 100
print("=" * 60)
print("CLAIM 1: Ambiguous zone (valence 4.0-6.0)")
print(f"  Result: {n_ambig}/{total} = {pct:.1f}%")
print(f"  Paper claims: 385/1280 = 30.1%")
print(f"  MATCH: {'YES' if abs(pct - 30.1) < 1 else 'NO — INVESTIGATE'}")
print()

# Also check ±0.25 around threshold=5.0
near_threshold = df[(df['Valence'] >= 4.75) & (df['Valence'] <= 5.25)]
print(f"  Near threshold (4.75-5.25): {len(near_threshold)}/{total} = {len(near_threshold)/total*100:.1f}%")
print(f"  Paper claims: ~15%")
print()

# ─── Claim 2: Per-subject calibration ───
print("=" * 60)
print("CLAIM 2: Per-subject mean valence")
per_subj = df.groupby('Participant_id')['Valence'].mean().sort_values()
print(f"  Subject with lowest mean: s{per_subj.index[0]:02d} = {per_subj.iloc[0]:.2f}")
print(f"  Subject with highest mean: s{per_subj.index[-1]:02d} = {per_subj.iloc[-1]:.2f}")
print(f"  Range: {per_subj.max() - per_subj.min():.2f} points")
print()

# Check specific subjects from paper claim
for sid in [4, 6]:
    if sid in per_subj.index:
        print(f"  s{sid:02d} mean valence: {per_subj[sid]:.2f}")
print()
print("  Paper claims: s04=4.12, s06=6.16")
print(f"  Full per-subject table:")
for sid in per_subj.index:
    print(f"    s{sid:02d}: {per_subj[sid]:.2f}")
print()

# ─── Claim 3: Majority baseline across thresholds ───
print("=" * 60)
print("CLAIM 3: Majority-class baseline across binarization thresholds")
print()
print(f"  {'Threshold':>12}  {'High':>6}  {'Low':>6}  {'Majority%':>10}  {'N_high':>7}  {'N_low':>6}")
print(f"  {'-'*12}  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*7}  {'-'*6}")

thresholds = [4.0, 4.5, 5.0, 5.25, 5.5, 6.0]
# Also test strict (>) vs non-strict (>=) for threshold=5.0
for t in thresholds:
    high = (df['Valence'] >= t).sum()
    low = (df['Valence'] < t).sum()
    majority = max(high, low) / total * 100
    print(f"  {'>='+str(t):>12}  {high:>6}  {low:>6}  {majority:>9.1f}%  {high:>7}  {low:>6}")

# Exact threshold the paper uses: <=5 means valence <= 5.0 is low
print()
print("  Alternative: threshold <=X means low")
for t in [4, 5]:
    low = (df['Valence'] <= t).sum()
    high = (df['Valence'] > t).sum()
    majority = max(high, low) / total * 100
    print(f"  {'<='+str(t):>12}  {high:>6}  {low:>6}  {majority:>9.1f}%  H={high:>5}  L={low:>5}")

print()
print("  Paper claims: 56.5% (threshold <=5) to 72.2% (threshold <=4)")
print()

# ─── Extra: Histogram of valence distribution ───
print("=" * 60)
print("EXTRA: Valence histogram (1-point bins)")
for low_edge in range(1, 10):
    high_edge = low_edge + 1
    count = ((df['Valence'] >= low_edge) & (df['Valence'] < high_edge)).sum()
    bar = '#' * (count // 5)
    print(f"  [{low_edge}-{high_edge}): {count:>4}  {bar}")

# ─── Extra: Arousal stats (for completeness) ───
print()
print("=" * 60)
print("EXTRA: Arousal ambiguous zone (4.0-6.0)")
arousal_ambig = df[(df['Arousal'] >= 4.0) & (df['Arousal'] <= 6.0)]
print(f"  {len(arousal_ambig)}/{total} = {len(arousal_ambig)/total*100:.1f}%")

per_subj_arousal = df.groupby('Participant_id')['Arousal'].mean()
print(f"  Per-subject arousal range: {per_subj_arousal.min():.2f} - {per_subj_arousal.max():.2f} ({per_subj_arousal.max()-per_subj_arousal.min():.2f} span)")
