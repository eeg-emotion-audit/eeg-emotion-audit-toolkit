"""
Build DEAP stimulus-vs-self-report mismatch tables for BOTH valence and arousal,
using the online normative ratings (DEAP_video_list.xls) as the stimulus-level label.

Mirrors the FACED 19.8% valence mismatch analysis (verify_faced_labels.py) — now extended
to arousal, which FACED cannot do without a derivation step.

Inputs (all in this directory):
    DEAP_video_list.xls     — 120 candidate videos × 25 cols, incl. AVG_Valence / AVG_Arousal
    online_ratings.xls      — 1778 raw ratings from online screening cohort
    participant_ratings.xls — per-trial per-subject SAM ratings (32 subj × 40 trials)

Outputs (printed to stdout + JSON dump):
    Mismatch rates at threshold 5.0 (DEAP standard) for valence AND arousal
    Per-clip stimulus-level binarization summary
    Krippendorff's alpha on the online cohort for both dimensions
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path

try:
    import krippendorff
    HAS_KRIPP = True
except ImportError:
    HAS_KRIPP = False
    print("[note] krippendorff not installed; skipping online-cohort alpha")

BASE = Path(__file__).parent
THRESH = 5.0

video_list = pd.read_excel(BASE / "DEAP_video_list.xls")
online = pd.read_excel(BASE / "online_ratings.xls")
pr = pd.read_excel(BASE / "participant_ratings.xls")

exp_clips = video_list.dropna(subset=["Experiment_id"]).copy()
exp_clips["Experiment_id"] = exp_clips["Experiment_id"].astype(int)
assert len(exp_clips) == 40, f"expected 40 experiment clips, got {len(exp_clips)}"

print(f"Participant ratings shape: {pr.shape}")
print(f"Participant ratings cols:  {list(pr.columns)}")
print(f"Experiment clips:          {len(exp_clips)}")
print(f"Online ratings total:      {len(online)}")

online_exp = online.merge(exp_clips[["Online_id", "Experiment_id"]], on="Online_id", how="inner")
print(f"Online ratings on exp 40:  {len(online_exp)}\n")

results = {"threshold": THRESH}

for dim in ["Valence", "Arousal"]:
    stim_map = dict(zip(exp_clips["Experiment_id"], exp_clips[f"AVG_{dim}"]))
    stim_bin = {eid: int(v >= THRESH) for eid, v in stim_map.items()}

    sr_raw = pr[dim].values.astype(float)
    stim_raw = np.array([stim_map[int(eid)] for eid in pr["Experiment_id"].values])
    sr_binarr = (sr_raw >= THRESH).astype(int)
    stim_binarr = (sr_raw >= THRESH).astype(int) * 0 + np.array([stim_bin[int(eid)] for eid in pr["Experiment_id"].values])
    mismatch = (sr_binarr != stim_binarr)

    print(f"=== {dim} ===")
    vals = np.array(list(stim_map.values()))
    print(f"  stim-level range         {vals.min():.2f}..{vals.max():.2f}  (mean {vals.mean():.2f}, std {vals.std():.2f})")
    print(f"  stim-high clips (>=5.0)  {sum(stim_bin.values())}/40")
    print(f"  SR range                 {sr_raw.min():.2f}..{sr_raw.max():.2f}  (mean {sr_raw.mean():.2f}, std {sr_raw.std():.2f})")
    print(f"  mismatch rate            {mismatch.mean()*100:.2f}%  ({mismatch.sum()}/{len(mismatch)})")

    # Ambiguous zone (SR in 4..6)
    amb = ((sr_raw >= 4.0) & (sr_raw <= 6.0)).sum()
    print(f"  SR ambiguous zone 4-6    {amb}/{len(sr_raw)} = {amb/len(sr_raw)*100:.2f}%")

    # Online-cohort alpha per clip (inter-rater reliability on stim-level label)
    if HAS_KRIPP:
        pivot = online_exp.pivot_table(index="Online_id", columns=online_exp.groupby("Online_id").cumcount(), values=dim)
        alpha = krippendorff.alpha(reliability_data=pivot.values, level_of_measurement="interval")
        print(f"  online-cohort alpha      {alpha:.4f}  (interval, per-clip across ~14-16 raters)")
    else:
        alpha = None

    results[dim.lower()] = {
        "stim_high_count": sum(stim_bin.values()),
        "mismatch_rate_pct": round(float(mismatch.mean() * 100), 3),
        "mismatch_count": int(mismatch.sum()),
        "total_trials": int(len(mismatch)),
        "sr_ambiguous_4_6_pct": round(float(amb / len(sr_raw) * 100), 3),
        "sr_mean": round(float(sr_raw.mean()), 3),
        "sr_std": round(float(sr_raw.std()), 3),
        "online_cohort_alpha": alpha,
    }
    print()

out_path = BASE / "deap_arousal_verify.json"
out_path.write_text(json.dumps(results, indent=2))
print(f"wrote {out_path}")
