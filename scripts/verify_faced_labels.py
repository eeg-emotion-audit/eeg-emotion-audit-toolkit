"""
Verify FACED label statistics for NeurIPS paper.
Claims to verify:
  1. 37.2% of trials in ambiguous zone
  2. 19.8% stimulus vs self-report mismatch (585/2952 non-neutral trials)
  3. Per-subject mean valence spans 3.74 points across 123 subjects
  4. Per-category mismatch: negative 10-16%, positive 23-31%
  5. Fear most consistent (90%), amusement least (68.6%)
"""
import scipy.io
import numpy as np
import os
from collections import defaultdict

FACED_DIR = os.environ.get('FACED_DIR', './data/faced')

# FACED stimulus-based emotion mapping (from TorchEEG, 0-indexed trials)
# Trials 0-2: anger, 3-5: disgust, 6-8: fear, 9-11: sadness
# 12-15: neutral (4 videos), 16-18: amusement, 19-21: inspiration
# 22-24: joy, 25-27: tenderness
# Map VIDEO ID (1-indexed, from vid field) to emotion
# Trials are presented in randomized order per subject — use vid, not trial index
VID_TO_EMOTION = {}
for v in range(1, 4): VID_TO_EMOTION[v] = 'anger'
for v in range(4, 7): VID_TO_EMOTION[v] = 'disgust'
for v in range(7, 10): VID_TO_EMOTION[v] = 'fear'
for v in range(10, 13): VID_TO_EMOTION[v] = 'sadness'
for v in range(13, 17): VID_TO_EMOTION[v] = 'neutral'
for v in range(17, 20): VID_TO_EMOTION[v] = 'amusement'
for v in range(20, 23): VID_TO_EMOTION[v] = 'inspiration'
for v in range(23, 26): VID_TO_EMOTION[v] = 'joy'
for v in range(26, 29): VID_TO_EMOTION[v] = 'tenderness'

STIMULUS_VALENCE = {
    'anger': 'negative', 'disgust': 'negative', 'fear': 'negative', 'sadness': 'negative',
    'neutral': 'neutral',
    'amusement': 'positive', 'inspiration': 'positive', 'joy': 'positive', 'tenderness': 'positive',
}

# Score indices in After_remarks.mat score array (0-indexed):
# 0=joy, 1=tenderness, 2=inspiration, 3=amusement, 4=anger, 5=disgust,
# 6=fear, 7=sadness, 8=arousal, 9=valence, 10=familiarity, 11=liking
VALENCE_IDX = 9
AROUSAL_IDX = 8

subjects = sorted([d for d in os.listdir(FACED_DIR) if d.startswith('sub')])
print(f"Found {len(subjects)} subjects")

all_valence = []
per_subject_mean_valence = {}
per_trial_data = []

for subj in subjects:
    mat_path = os.path.join(FACED_DIR, subj, 'After_remarks.mat')
    if not os.path.exists(mat_path):
        print(f"  WARNING: Missing {mat_path}")
        continue

    mat = scipy.io.loadmat(mat_path)
    ar = mat['After_remark']

    subj_valences = []
    for trial_idx in range(28):
        row = ar[trial_idx, 0]
        score = row['score']
        if score.size == 0:
            continue
        valence = float(score[0, VALENCE_IDX])
        arousal = float(score[0, AROUSAL_IDX])

        vid = int(row['vid'][0, 0])
        emotion = VID_TO_EMOTION[vid]
        stim_valence = STIMULUS_VALENCE[emotion]

        # Self-report binary (midpoint = 3.5 on 0-7 scale)
        sr_valence = 'positive' if valence >= 3.5 else 'negative'

        per_trial_data.append({
            'subject': subj,
            'trial': trial_idx,
            'emotion': emotion,
            'stim_valence': stim_valence,
            'sr_valence_raw': valence,
            'sr_arousal_raw': arousal,
            'sr_valence_binary': sr_valence,
        })

        subj_valences.append(valence)
        all_valence.append(valence)

    if subj_valences:
        per_subject_mean_valence[subj] = np.mean(subj_valences)

all_valence = np.array(all_valence)
print(f"Total trials with valence data: {len(all_valence)}")
print(f"Valence range: {all_valence.min():.2f} - {all_valence.max():.2f}")
print(f"Valence mean: {np.mean(all_valence):.2f}, std: {np.std(all_valence):.2f}")
print()

# ─── Claim 1: Ambiguous zone ───
# FACED scale is 0-7, midpoint is 3.5
# Ambiguous zone: define as within ±1.0 of midpoint = [2.5, 4.5]
# (matching the DEAP convention of ±1 around midpoint)
print("=" * 60)
print("CLAIM 1: FACED ambiguous zone")

# Try different zone definitions to find the 37.2%
zones = [
    ("±1.0 of midpoint [2.5, 4.5]", 2.5, 4.5),
    ("±1.5 of midpoint [2.0, 5.0]", 2.0, 5.0),
    ("±0.75 of midpoint [2.75, 4.25]", 2.75, 4.25),
]
for desc, lo, hi in zones:
    n = ((all_valence >= lo) & (all_valence <= hi)).sum()
    pct = n / len(all_valence) * 100
    print(f"  {desc}: {n}/{len(all_valence)} = {pct:.1f}%")

# The DEAP definition was 4.0-6.0 on a 1-9 scale (±1.0 around midpoint 5.0)
# Equivalent on 0-7 scale: midpoint 3.5, ±(7/9)*1.0 ≈ ±0.78, so [2.72, 4.28]
# Or simply scale proportionally: DEAP [4,6] = [44.4%, 66.7%] of range
# On 0-7 scale: [0.444*7, 0.667*7] = [3.11, 4.67]
scaled_lo = 4.0 / 9.0 * 7.0
scaled_hi = 6.0 / 9.0 * 7.0
n = ((all_valence >= scaled_lo) & (all_valence <= scaled_hi)).sum()
pct = n / len(all_valence) * 100
print(f"  Proportionally scaled from DEAP [{scaled_lo:.2f}, {scaled_hi:.2f}]: {n}/{len(all_valence)} = {pct:.1f}%")
print()
print(f"  Paper claims: 37.2%")
print()

# ─── Claim 2: Stimulus vs self-report mismatch ───
print("=" * 60)
print("CLAIM 2: Stimulus vs self-report valence mismatch")

non_neutral = [d for d in per_trial_data if d['stim_valence'] != 'neutral']
total_non_neutral = len(non_neutral)

mismatches = [d for d in non_neutral if d['stim_valence'] != d['sr_valence_binary']]
n_mismatch = len(mismatches)
pct_mismatch = n_mismatch / total_non_neutral * 100

print(f"  Non-neutral trials: {total_non_neutral}")
print(f"  Mismatches: {n_mismatch}/{total_non_neutral} = {pct_mismatch:.1f}%")
print(f"  Paper claims: 585/2952 = 19.8%")
print(f"  MATCH: {'YES' if abs(pct_mismatch - 19.8) < 1 else 'NO — INVESTIGATE'}")
print()

# ─── Claim 2b: Per-category mismatch ───
print("  Per-category mismatch rates:")
categories = ['anger', 'disgust', 'fear', 'sadness', 'amusement', 'inspiration', 'joy', 'tenderness']
for cat in categories:
    cat_trials = [d for d in non_neutral if d['emotion'] == cat]
    cat_mismatches = [d for d in cat_trials if d['stim_valence'] != d['sr_valence_binary']]
    if cat_trials:
        consistency = 1 - len(cat_mismatches) / len(cat_trials)
        print(f"    {cat:>12}: {len(cat_mismatches):>3}/{len(cat_trials):>3} mismatch = {len(cat_mismatches)/len(cat_trials)*100:.1f}% (consistency: {consistency*100:.1f}%)")

print()
print("  Paper claims: negative 10-16% mismatch, positive 23-31%")
print("  Paper claims: fear most consistent (90%), amusement least (68.6%)")
print()

# ─── Claim 3: Per-subject mean valence span ───
print("=" * 60)
print("CLAIM 3: Per-subject mean valence span")

means = np.array(list(per_subject_mean_valence.values()))
subjects_sorted = sorted(per_subject_mean_valence.items(), key=lambda x: x[1])

print(f"  Lowest mean: {subjects_sorted[0][0]} = {subjects_sorted[0][1]:.2f}")
print(f"  Highest mean: {subjects_sorted[-1][0]} = {subjects_sorted[-1][1]:.2f}")
span = subjects_sorted[-1][1] - subjects_sorted[0][1]
print(f"  Span: {span:.2f} points")
print(f"  Paper claims: 3.74 points")
print(f"  MATCH: {'YES' if abs(span - 3.74) < 0.1 else 'NO — INVESTIGATE'}")
print()
print(f"  Mean of means: {np.mean(means):.2f}")
print(f"  Std of means: {np.std(means):.2f}")
print()

# ─── Extra: Distribution of self-reported valence ───
print("=" * 60)
print("EXTRA: Self-reported valence histogram (0.5-point bins)")
for low_edge_x10 in range(0, 70, 5):
    low_edge = low_edge_x10 / 10.0
    high_edge = low_edge + 0.5
    count = ((all_valence >= low_edge) & (all_valence < high_edge)).sum()
    bar = '#' * (count // 10)
    print(f"  [{low_edge:.1f}-{high_edge:.1f}): {count:>4}  {bar}")

# ─── Extra: Majority baseline for FACED self-report binary ───
print()
print("=" * 60)
print("EXTRA: FACED self-report majority baselines")
for t in [3.0, 3.5, 4.0]:
    high = (all_valence >= t).sum()
    low = (all_valence < t).sum()
    majority = max(high, low) / len(all_valence) * 100
    print(f"  Threshold {t}: high={high}, low={low}, majority={majority:.1f}%")
