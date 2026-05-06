"""
DREAMER stimulus-vs-self-report mismatch verification.

Analog to `scripts/verify_deap_arousal.py`, but DREAMER has no separate
normative cohort like DEAP's `video_list.xls`. Stimulus-level labels are
derived from Katsigiannis 2018 Table I target-emotion categories mapped
to Russell quadrants. Self-report labels are per-subject SAM 1-5 integer,
binarized strict >3 per DREAMER's own convention (Katsigiannis 2018
Table III reproduces exactly with this rule — 3/4/6/5 = LALV/LAHV/HALV/HAHV).

DREAMER paper facts:
    - 18 film clips, 23 final subjects (2 excluded of 25 recruited).
    - Target emotions: calmness, surprise, amusement, fear, excitement,
      disgust, happiness, anger, sadness. Two clips per emotion.
    - Stimulus catalog: Gabert-Quillen et al. 2008 (NOT Gross-Levenson 1995).
    - SAM 1-5 discrete integer scale on V/A/D.
    - EEG: 14-channel Emotiv EPOC, 128 Hz, 4-30 Hz bandpass.
    - Last 60s of each clip used for feature extraction.

Outputs (printed + JSON dump):
    - Per-dimension (V, A) mismatch rate on 23x18 = 414 trials.
    - Fleiss kappa on SR binarization (inter-subject, per clip).
    - Per-emotion-category breakdown: mismatch% + mean SAM for V and A.
"""
import os
import json
from pathlib import Path

import numpy as np
import scipy.io

BASE = Path(__file__).parent
DREAMER_MAT = os.environ.get('DREAMER_MAT', './data/dreamer/DREAMER.mat')
THRESH = 3  # strict >3 HIGH, <=3 LOW, per Katsigiannis 2018 Table III convention

# Katsigiannis 2018 Table I — per-clip target emotion (ID 1..18 in paper order).
# Two clips per emotion, 9 emotions total.
CATEGORY_PER_CLIP = [
    "calmness",    # 1  Searching for Bobby Fischer
    "surprise",    # 2  D.O.A.
    "amusement",   # 3  The Hangover
    "fear",        # 4  The Ring
    "excitement",  # 5  300
    "disgust",     # 6  National Lampoon's VanWilder
    "happiness",   # 7  Wall-E
    "anger",       # 8  Crash
    "sadness",     # 9  My Girl
    "disgust",     # 10 The Fly
    "calmness",    # 11 Pride and Prejudice
    "amusement",   # 12 Modern Times
    "happiness",   # 13 Remember the Titans
    "anger",       # 14 Gentlemans Agreement
    "fear",        # 15 Psycho
    "excitement",  # 16 The Bourne Identity
    "sadness",     # 17 The Shawshank Redemption
    "surprise",    # 18 The Departed
]
assert len(CATEGORY_PER_CLIP) == 18

# Russell-circumplex category -> binary (V_high, A_high).
# Calibration against Table I mean ratings (strict >3):
#   agreement on 14/18 clips, disagreement on clips 12 (amusement, A=2.61),
#   14 (anger, A=2.22), 2 (surprise, V=3.04), 11 (calmness, A=1.96, agrees).
# Category map uses circumplex convention; Table I numeric disagreement is the
# expected phenomenon the §4.5 mismatch framing exists to characterize.
# Ambiguous positions resolved per the per-category Russell mapping notes:
#   surprise -> V=LOW (DREAMER clips empirically negative per Table I),
#              A=HIGH (Russell canonical startle).
#   happiness -> V=HIGH, A=HIGH (excited-joy quadrant per Russell; not
#              'serene contentment' which is calmness).
RUSSELL_MAP = {
    "amusement":  {"V": 1, "A": 1},
    "excitement": {"V": 1, "A": 1},
    "happiness":  {"V": 1, "A": 1},
    "calmness":   {"V": 1, "A": 0},
    "anger":      {"V": 0, "A": 1},
    "disgust":    {"V": 0, "A": 1},
    "fear":       {"V": 0, "A": 1},
    "sadness":    {"V": 0, "A": 0},
    "surprise":   {"V": 0, "A": 1},
}
assert set(CATEGORY_PER_CLIP).issubset(RUSSELL_MAP.keys())


def fleiss_kappa_binary(binary_matrix):
    """Fleiss' kappa on (n_raters, n_items) binary matrix."""
    n_raters, n_items = binary_matrix.shape
    n_pos = binary_matrix.sum(axis=0)
    n_neg = n_raters - n_pos
    table = np.stack([n_neg, n_pos], axis=1).astype(np.float64)  # (items, cats)
    n = table.sum(axis=1)
    assert np.allclose(n, n_raters), "all items must have same rater count"
    Pi = (np.sum(table ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = Pi.mean()
    p_j = table.sum(axis=0) / (n_items * n_raters)
    P_e = np.sum(p_j ** 2)
    if np.isclose(1.0, P_e):
        return 1.0
    return float((P_bar - P_e) / (1.0 - P_e))


def main():
    mat = scipy.io.loadmat(DREAMER_MAT, squeeze_me=True, struct_as_record=False)
    dr = mat["DREAMER"]
    n_sub = int(dr.noOfSubjects)
    n_vid = int(dr.noOfVideoSequences)
    assert n_sub == 23 and n_vid == 18, f"expected 23x18, got {n_sub}x{n_vid}"

    # Stack SAM scores: (23, 18)
    V_sr = np.stack([s.ScoreValence for s in dr.Data]).astype(np.int64)
    A_sr = np.stack([s.ScoreArousal for s in dr.Data]).astype(np.int64)
    assert V_sr.shape == (23, 18) and A_sr.shape == (23, 18)
    assert V_sr.min() >= 1 and V_sr.max() <= 5
    assert A_sr.min() >= 1 and A_sr.max() <= 5

    # Per-clip stimulus binary labels from category
    stim_V = np.array([RUSSELL_MAP[c]["V"] for c in CATEGORY_PER_CLIP], dtype=np.int64)
    stim_A = np.array([RUSSELL_MAP[c]["A"] for c in CATEGORY_PER_CLIP], dtype=np.int64)
    assert stim_V.shape == (18,) and stim_A.shape == (18,)

    # Binarize SR: strict >3 HIGH, else LOW
    V_sr_bin = (V_sr > THRESH).astype(np.int64)   # (23, 18)
    A_sr_bin = (A_sr > THRESH).astype(np.int64)

    results = {
        "dataset": "DREAMER",
        "source": DREAMER_MAT,
        "threshold": THRESH,
        "binarization": f"sr > {THRESH} HIGH, sr <= {THRESH} LOW (per Katsigiannis 2018 Table III)",
        "scale_type": "discrete_5point",
        "stimulus_source": "Katsigiannis 2018 Table I target emotions, Russell-circumplex map (category -> HIGH/LOW per RUSSELL_MAP)",
        "stimulus_catalog_citation": "Gabert-Quillen et al. 2008 (ref [21] in Katsigiannis 2018)",
        "n_subjects": n_sub,
        "n_clips": n_vid,
        "n_trials_total": n_sub * n_vid,
    }

    print(f"=== DREAMER verify (threshold strict >{THRESH}) ===")
    print(f"{n_sub} subjects x {n_vid} clips = {n_sub * n_vid} trials")
    print()

    for dim_name, sr_raw, sr_bin, stim_bin in [
        ("valence", V_sr, V_sr_bin, stim_V),
        ("arousal", A_sr, A_sr_bin, stim_A),
    ]:
        # Broadcast stim_bin (18,) across 23 subjects for comparison.
        stim_per_trial = np.broadcast_to(stim_bin, sr_bin.shape)
        mismatch = sr_bin != stim_per_trial
        mm_rate = mismatch.mean()
        kappa = fleiss_kappa_binary(sr_bin)

        # Ambiguous zone: SR == 3 exactly (mid-scale tie → LOW under strict >3)
        amb_eq3 = (sr_raw == 3).sum()

        dim_res = {
            "stim_high_count": int(stim_bin.sum()),
            "stim_low_count": int((1 - stim_bin).sum()),
            "mismatch_rate_pct": round(float(mm_rate * 100), 3),
            "mismatch_count": int(mismatch.sum()),
            "total_trials": int(mismatch.size),
            "sr_mean": round(float(sr_raw.mean()), 3),
            "sr_std": round(float(sr_raw.std()), 3),
            "sr_equals_3_count_pct": round(float(amb_eq3 / sr_raw.size * 100), 3),
            "fleiss_kappa_inter_subject": round(kappa, 4),
        }
        results[dim_name] = dim_res

        print(f"-- {dim_name} --")
        print(f"  stim HIGH count          {dim_res['stim_high_count']}/18  LOW {dim_res['stim_low_count']}/18")
        print(f"  mismatch rate            {dim_res['mismatch_rate_pct']:.2f}%  ({dim_res['mismatch_count']}/{dim_res['total_trials']})")
        print(f"  SR range                 {sr_raw.min()}..{sr_raw.max()}  (mean {dim_res['sr_mean']:.2f}, std {dim_res['sr_std']:.2f})")
        print(f"  SR == 3 ties (fall LOW)  {dim_res['sr_equals_3_count_pct']:.2f}%")
        print(f"  Fleiss kappa (inter-sub) {dim_res['fleiss_kappa_inter_subject']:.4f}")
        print()

    # Per-category breakdown
    categories = sorted(set(CATEGORY_PER_CLIP))
    per_cat = {}
    for cat in categories:
        clip_idx = [i for i, c in enumerate(CATEGORY_PER_CLIP) if c == cat]
        n_clips_cat = len(clip_idx)

        # Valence mismatch in this category
        v_mm = (V_sr_bin[:, clip_idx] != stim_V[clip_idx][None, :])
        a_mm = (A_sr_bin[:, clip_idx] != stim_A[clip_idx][None, :])

        per_cat[cat] = {
            "n_clips": n_clips_cat,
            "clip_ids_1based": [i + 1 for i in clip_idx],
            "russell_stim_V": int(RUSSELL_MAP[cat]["V"]),
            "russell_stim_A": int(RUSSELL_MAP[cat]["A"]),
            "mean_sr_V": round(float(V_sr[:, clip_idx].mean()), 3),
            "mean_sr_A": round(float(A_sr[:, clip_idx].mean()), 3),
            "valence_mismatch_pct": round(float(v_mm.mean() * 100), 3),
            "arousal_mismatch_pct": round(float(a_mm.mean() * 100), 3),
            "n_trials": int(v_mm.size),
        }

    results["per_category"] = per_cat

    print("-- per-category breakdown --")
    print(f"{'category':<12} {'n_clip':>6} {'R_V':>4} {'R_A':>4} "
          f"{'mean_V':>7} {'mean_A':>7} {'V_mm%':>6} {'A_mm%':>6}")
    for cat, d in per_cat.items():
        print(f"{cat:<12} {d['n_clips']:>6} {d['russell_stim_V']:>4} {d['russell_stim_A']:>4} "
              f"{d['mean_sr_V']:>7.2f} {d['mean_sr_A']:>7.2f} "
              f"{d['valence_mismatch_pct']:>6.2f} {d['arousal_mismatch_pct']:>6.2f}")

    out = BASE / "dreamer_arousal_verify.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
