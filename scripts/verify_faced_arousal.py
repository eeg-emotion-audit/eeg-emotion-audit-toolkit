#!/usr/bin/env python3
"""FACED stimulus-vs-self-report arousal mismatch verification.

Extends the valence mismatch in verify_faced_labels.py to the arousal axis.
Uses the Russell circumplex mapping on the 9 emotion categories with two
bracket variants for the ambiguous 'inspiration' category.

Reproduces the 39.5-43.1% arousal mismatch sensitivity range cited in
Section 4.4 of the paper, plus the Krippendorff alpha values for arousal.

Inputs:
  --faced-dir   FACED root with sub*/After_remarks.mat (default: ./data/faced)
  --output      Output JSON path (default: ./results/faced_arousal_verify.json)
  --threshold   Binarization threshold on the 0-7 SAM scale, inclusive (default: 3.5)
"""
import argparse
import json
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


def run(faced_dir, out_json, threshold):
    from scipy.io import loadmat

    subjects = sorted([d for d in os.listdir(faced_dir) if d.startswith("sub")])
    rows = []
    per_clip_sr_bin = {}

    for subj in subjects:
        mat_path = os.path.join(faced_dir, subj, "After_remarks.mat")
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
                continue
            ar = float(sc[AROUSAL_IDX])
            va = float(sc[VALENCE_IDX])
            sr_bin = int(ar >= threshold)
            rows.append({
                "subj": subj, "vid": vid, "emotion": emotion,
                "sr_arousal": ar, "sr_valence": va, "sr_bin": sr_bin,
            })
            per_clip_sr_bin.setdefault(vid, []).append(sr_bin)

    results = {"threshold": threshold, "n_trials": len(rows), "n_subjects": len(subjects)}

    for variant_name, mapping in [
        ("inspiration_low",  STIM_AROUSAL_INSP_LOW),
        ("inspiration_high", STIM_AROUSAL_INSP_HIGH),
    ]:
        n_mm = sum(1 for r in rows if r["sr_bin"] != mapping[r["emotion"]])
        rate = n_mm / len(rows) if rows else float("nan")
        per_cat = {}
        for emo in set(r["emotion"] for r in rows):
            emo_rows = [r for r in rows if r["emotion"] == emo]
            mm = sum(1 for r in emo_rows if r["sr_bin"] != mapping[emo])
            per_cat[emo] = {
                "n": len(emo_rows),
                "stim_arousal": "HIGH" if mapping[emo] == 1 else "LOW",
                "mismatch_pct": round(100 * mm / len(emo_rows), 2),
                "mean_sr_arousal": round(float(np.mean([r["sr_arousal"] for r in emo_rows])), 3),
            }
        results[variant_name] = {
            "stim_high_clips": sum(1 for v, e in VID_TO_EMOTION.items() if e != "neutral" and mapping[e] == 1),
            "stim_low_clips":  sum(1 for v, e in VID_TO_EMOTION.items() if e != "neutral" and mapping[e] == 0),
            "mismatch_count": n_mm,
            "mismatch_rate_pct": round(100 * rate, 3),
            "per_category": per_cat,
        }

    try:
        import krippendorff
        clips = sorted(per_clip_sr_bin.keys())
        max_sub = max(len(v) for v in per_clip_sr_bin.values())
        mat = np.full((len(clips), max_sub), np.nan)
        for i, c in enumerate(clips):
            vals = per_clip_sr_bin[c]
            mat[i, :len(vals)] = vals
        results["alpha_sr_arousal_binary_nominal"] = round(
            krippendorff.alpha(reliability_data=mat.T, level_of_measurement="nominal"), 4)
        results["alpha_sr_arousal_binary_interval"] = round(
            krippendorff.alpha(reliability_data=mat.T, level_of_measurement="interval"), 4)

        per_clip_raw = {}
        for r in rows:
            per_clip_raw.setdefault(r["vid"], []).append(r["sr_arousal"])
        mat_raw = np.full((len(clips), max_sub), np.nan)
        for i, c in enumerate(clips):
            vals = per_clip_raw[c]
            mat_raw[i, :len(vals)] = vals
        results["alpha_sr_arousal_continuous_interval"] = round(
            krippendorff.alpha(reliability_data=mat_raw.T, level_of_measurement="interval"), 4)
    except ImportError:
        results["alpha_note"] = "krippendorff not installed; skipped"

    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"=== FACED arousal stim-vs-SR mismatch (tau={threshold}, 24 non-neutral clips) ===")
    print(f"n_trials={results['n_trials']}, n_subjects={results['n_subjects']}")
    for variant in ("inspiration_low", "inspiration_high"):
        r = results[variant]
        print(f"[{variant}] stim HIGH={r['stim_high_clips']} / LOW={r['stim_low_clips']}  "
              f"mismatch={r['mismatch_rate_pct']}% ({r['mismatch_count']}/{results['n_trials']})")
    if "alpha_sr_arousal_continuous_interval" in results:
        print(f"alpha SR arousal continuous (interval): {results['alpha_sr_arousal_continuous_interval']}")
        print(f"alpha SR arousal binary    (nominal):   {results['alpha_sr_arousal_binary_nominal']}")
    print(f"\nwrote {out_json}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--faced-dir", default="./data/faced",
                        help="FACED root with sub*/After_remarks.mat (default: %(default)s)")
    parser.add_argument("--output", default="./results/faced_arousal_verify.json",
                        help="Output JSON path (default: %(default)s)")
    parser.add_argument("--threshold", type=float, default=3.5,
                        help="Binarization threshold (inclusive) on the 0-7 SAM scale")
    args = parser.parse_args()
    run(args.faced_dir, args.output, args.threshold)


if __name__ == "__main__":
    main()
