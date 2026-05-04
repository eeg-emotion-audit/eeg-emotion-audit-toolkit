#!/usr/bin/env python3
"""All-32-subject companion to phantom_delta_spectrum.py.

The existing paper figure (figures/phantom_delta_psd.pdf) and the 0.17x ratio
cited in App I are computed across 4 representative subjects (s01, s10, s20, s32)
in phantom_delta_spectrum.py. This script computes the same delta/theta band-power
ratio across all 32 DEAP subjects for robustness verification, without
overwriting the existing figure or modifying the original script.

Output: prints band-power table + delta/theta ratio for the full 32-subject set.
Optionally saves a separate figure phantom_delta_psd_all32.pdf.
"""
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch

DATA_DIR = os.environ.get('DEAP_DIR', './data/deap')
OUT_DIR = os.environ.get('OUTPUT_DIR', './output/figures')
os.makedirs(OUT_DIR, exist_ok=True)

FS = 128

subjects_to_compute = list(range(1, 33))
band_defs = {
    'delta (1-4 Hz)': (1, 4),
    'theta (4-8 Hz)': (4, 8),
    'alpha (8-14 Hz)': (8, 14),
    'beta (14-31 Hz)': (14, 31),
    'gamma (31-45 Hz)': (31, 45),
}

band_powers_all = {b: [] for b in band_defs}
per_subject_ratios = []

for subj in subjects_to_compute:
    fname = os.path.join(DATA_DIR, f's{subj:02d}.dat')
    with open(fname, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    eeg = data['data'][:, :32, :]
    eeg_concat = eeg.reshape(-1, eeg.shape[-1])

    freqs, psd = welch(eeg_concat, fs=FS, nperseg=256, noverlap=128)
    psd_mean = psd.mean(axis=0)

    subj_band_powers = {}
    for bname, (f_lo, f_hi) in band_defs.items():
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        p = psd_mean[mask].mean()
        band_powers_all[bname].append(p)
        subj_band_powers[bname] = p

    per_subject_ratios.append(subj_band_powers['delta (1-4 Hz)'] / subj_band_powers['theta (4-8 Hz)'])

print("\n=== Band Power Analysis: ALL 32 DEAP subjects ===\n")
print(f"{'Band':<25} {'Mean PSD':>12} {'Ratio to theta':>16}")
print("-" * 55)
theta_power = np.mean(band_powers_all['theta (4-8 Hz)'])
for bname, powers in band_powers_all.items():
    mean_p = np.mean(powers)
    ratio = mean_p / theta_power
    marker = " *** PHANTOM" if 'delta' in bname else ""
    print(f"{bname:<25} {mean_p:>12.4f} {ratio:>16.4f}{marker}")

print(f"\nDelta/theta power ratio (mean of subject-level means): "
      f"{np.mean(band_powers_all['delta (1-4 Hz)'])/theta_power:.4f}")

per_subj_arr = np.array(per_subject_ratios)
print(f"Delta/theta per-subject ratio: mean={per_subj_arr.mean():.4f}, "
      f"std={per_subj_arr.std():.4f}, "
      f"min={per_subj_arr.min():.4f}, max={per_subj_arr.max():.4f}, n={len(per_subj_arr)}")

print("\n=== Comparison with 4-subject paper number ===")
paper_subjects = [1, 10, 20, 32]
paper_indices = [s - 1 for s in paper_subjects]
paper_delta = np.mean([band_powers_all['delta (1-4 Hz)'][i] for i in paper_indices])
paper_theta = np.mean([band_powers_all['theta (4-8 Hz)'][i] for i in paper_indices])
print(f"4-subj (s01, s10, s20, s32) delta/theta ratio: {paper_delta/paper_theta:.4f} "
      f"(should match App I caption '0.17x')")
