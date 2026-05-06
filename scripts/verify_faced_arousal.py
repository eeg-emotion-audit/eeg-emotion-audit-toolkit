"""
FACED stim-vs-SR arousal mismatch verification.

Parallels verify_deap_arousal.py. Extends verify_faced_labels.py to the arousal
dimension using the Russell circumplex mapping on the 9 emotion categories
(see paper Appendix on stimulus-arousal mapping for the per-category derivation).

Inputs:
    data/faced/sub{NNN}/After_remarks.mat
        per-subject 28-clip SAM scores (score vector index 8 = arousal, 9 = valence)

Convention (matches FACED valence convention for paired comparability):
    - SR arousal binarized at >= 3.5 (inclusive) on the 0-7 SAM scale.
    - Neutral-excluded: 24 non-neutral clips retained (4 neutral clips dropped
      for symmetry with the FACED binary-classification protocol).

Stimulus-level arousal — two bracket variants for Inspiration (ambiguous per
the paper Appendix on stimulus-arousal mapping):
    Inspiration-LOW:  HIGH = {amusement, joy, anger, disgust, fear} (15 clips)
                      LOW  = {sadness, tenderness, inspiration}    ( 9 clips)
    Inspiration-HIGH: HIGH = {amusement, joy, inspiration, anger, disgust, fear} (18)
                      LOW  = {sadness, tenderness}                              ( 6)

Outputs:
    faced_arousal_verify.json — per-variant mismatch stats + Fleiss κ
"""
import os, json
import numpy as np
from scipy.io import loadmat

FACED_DIR = os.environ.get('FACED_DIR', './data/faced')
OUT_JSON = os.path.join(os.path.dirname(__file__), "..", "faced_samples", "faced_arousal_verify.json")

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

STIM_AROUSAL_INSP_LOW = {
    "anger": 1, "disgust": 1, "fear": 1, "amusement": 1, "joy": 1,
    "sadness": 0, "tenderness": 0, "inspiration": 0,
}
STIM_AROUSAL_INSP_HIGH = {
    "anger": 1, "disgust": 1, "fear": 1, "amusement": 1, "joy": 1, "inspiration": 1,
    "sadness": 0, "tenderness": 0,
}

AROUSAL_IDX = 8
VALENCE_IDX = 9
THRESH = 3.5  # FACED convention, inclusive

def run():
    subjects = sorted([d for d in os.listdir(FACED_DIR) if d.startswith("sub")])
    rows = []  # per non-neutral trial
    # Also build per-clip SR-arousal-binary matrix for Fleiss κ (123 × 24)
    per_clip_sr_bin = {}

    for s_idx, subj in enumerate(subjects):
        mat_path = os.path.join(FACED_DIR, subj, "After_remarks.mat")
        if not os.path.exists(mat_path):
            continue
        m = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        entries = m["After_remark"]
        for e in entries:
            sc = e.score
            if sc is None or np.ndim(sc) == 0 or np.size(sc) == 0:
                continue
            vid = int(e.vid)
            emotion = VID_TO_EMOTION[vid]
            if emotion == "neutral":
                continue  # exclude neutral for symmetry with the binary-classification protocol
            ar = float(sc[AROUSAL_IDX])
            va = float(sc[VALENCE_IDX])
            sr_bin = int(ar >= THRESH)
            rows.append({
                "subj": subj, "vid": vid, "emotion": emotion,
                "sr_arousal": ar, "sr_valence": va, "sr_bin": sr_bin,
            })
            per_clip_sr_bin.setdefault(vid, []).append(sr_bin)

    # Compute mismatch per variant
    results = {"threshold": THRESH, "n_trials": len(rows), "n_subjects": len(subjects)}

    for variant_name, mapping in [
        ("inspiration_low",  STIM_AROUSAL_INSP_LOW),
        ("inspiration_high", STIM_AROUSAL_INSP_HIGH),
    ]:
        n_mm = sum(1 for r in rows if r["sr_bin"] != mapping[r["emotion"]])
        rate = n_mm / len(rows) if rows else float("nan")
        # Per-emotion-category mismatch
        per_cat = {}
        for emo in set(r["emotion"] for r in rows):
            emo_rows = [r for r in rows if r["emotion"] == emo]
            mm = sum(1 for r in emo_rows if r["sr_bin"] != mapping[emo])
            per_cat[emo] = {
                "n": len(emo_rows),
                "stim_arousal": "HIGH" if mapping[emo] == 1 else "LOW",
                "mismatch_pct": round(100 * mm / len(emo_rows), 2),
                "mean_sr_arousal": round(np.mean([r["sr_arousal"] for r in emo_rows]), 3),
            }
        results[variant_name] = {
            "stim_high_clips": sum(1 for v, e in VID_TO_EMOTION.items() if e != "neutral" and mapping[e] == 1),
            "stim_low_clips":  sum(1 for v, e in VID_TO_EMOTION.items() if e != "neutral" and mapping[e] == 0),
            "mismatch_count": n_mm,
            "mismatch_rate_pct": round(100 * rate, 3),
            "per_category": per_cat,
        }

    # Fleiss κ on SR arousal (123 subjects × 24 clips)
    try:
        import krippendorff
        # matrix: rows = clips, cols = subject raters, values = binary sr_arousal
        clips = sorted(per_clip_sr_bin.keys())
        max_sub = max(len(v) for v in per_clip_sr_bin.values())
        mat = np.full((len(clips), max_sub), np.nan)
        for i, c in enumerate(clips):
            vals = per_clip_sr_bin[c]
            mat[i, :len(vals)] = vals
        # Krippendorff expects shape (coders, units) = (subjects, clips)
        alpha_nom = krippendorff.alpha(reliability_data=mat.T, level_of_measurement="nominal")
        alpha_int = krippendorff.alpha(reliability_data=mat.T, level_of_measurement="interval")
        # Continuous-scale Fleiss κ equivalent via κ interval
        results["kappa_sr_arousal_binary_nominal"] = round(alpha_nom, 4)
        results["kappa_sr_arousal_binary_interval"] = round(alpha_int, 4)

        # Also α on raw continuous arousal (comparable to DEAP interval κ)
        per_clip_raw = {}
        for r in rows:
            per_clip_raw.setdefault(r["vid"], []).append(r["sr_arousal"])
        mat_raw = np.full((len(clips), max_sub), np.nan)
        for i, c in enumerate(clips):
            vals = per_clip_raw[c]
            mat_raw[i, :len(vals)] = vals
        alpha_raw = krippendorff.alpha(reliability_data=mat_raw.T, level_of_measurement="interval")
        results["kappa_sr_arousal_continuous_interval"] = round(alpha_raw, 4)
    except ImportError:
        results["kappa_note"] = "krippendorff not installed; skipped"

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"=== FACED arousal stim-vs-SR mismatch (τ={THRESH}, 24 non-neutral clips) ===")
    print(f"n_trials={results['n_trials']}, n_subjects={results['n_subjects']}")
    for variant in ("inspiration_low", "inspiration_high"):
        r = results[variant]
        print(f"[{variant}] stim HIGH={r['stim_high_clips']} / LOW={r['stim_low_clips']}  "
              f"mismatch={r['mismatch_rate_pct']}% ({r['mismatch_count']}/{results['n_trials']})")
    if "kappa_sr_arousal_continuous_interval" in results:
        print(f"κ SR arousal continuous (interval): {results['kappa_sr_arousal_continuous_interval']}")
        print(f"κ SR arousal binary    (nominal):   {results['kappa_sr_arousal_binary_nominal']}")
    print(f"\nwrote {OUT_JSON}")

if __name__ == "__main__":
    run()
