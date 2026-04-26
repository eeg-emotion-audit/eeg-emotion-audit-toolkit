# EEG-Emotion Benchmark Audit Toolkit

Companion code and data for an audit of EEG-based emotion-recognition
benchmarks across three independent layers of methodological variation:
**data preprocessing**, **label construction**, and **evaluation protocol**.

The toolkit packages:

- **A 40-paper, 15-dimension protocol survey** (`data/survey_table.csv`)
  spanning DEAP, FACED, SEED, and SEED-IV.
- **Seven self-contained audit scripts** (`scripts/`) that reproduce the
  numerical claims made in the paper from publicly available datasets:
  cross-pipeline DE correlation, Krippendorff's alpha for self-report
  reliability, stimulus-vs-self-report noise transition matrices, label
  threshold sensitivity, and the phantom-delta-band PSD verification.
- **A reporting checklist** (`CHECKLIST.md`) capturing the eight axes that
  must be disclosed for a result to be reproducible.
- **A Model Card template** (`MODELCARD_TEMPLATE.md`) extending the Mitchell
  et al. (2019) and Kinahan et al. (2024) frameworks with three EEG-emotion-
  specific field groups.

The toolkit is designed to be runnable on a laptop in under five minutes per
script once the underlying datasets are obtained.

## Quickstart

```bash
git clone <REPO_URL> eeg-emotion-audit-toolkit
cd eeg-emotion-audit-toolkit
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then point the scripts at your local copies of the datasets. Default paths
all live under `./data/`; override with `--faced-dir`, `--deap-ratings`,
etc. Pass `--help` to any script for the full argument list.

```bash
# Example: reproduce Table 3 (Krippendorff's alpha)
python scripts/compute_krippendorff.py \
    --faced-dir   ./data/faced \
    --deap-ratings ./data/deap/participant_ratings.csv

# Example: reproduce Section 4.4 (noise transition matrix on FACED)
python scripts/compute_noise_matrix.py --faced-dir ./data/faced

# Example: reproduce Section 4.1 (DEAP threshold sensitivity)
python scripts/verify_deap_labels.py --deap-ratings ./data/deap/participant_ratings.csv

# Example: reproduce Appendix E (phantom delta-band PSD)
python scripts/phantom_delta_spectrum.py \
    --data-dir ./data/deap_preprocessed_python \
    --out-dir  ./results/figures
```

Reference outputs from the paper's runs are committed in `results/`; reproducing
the cited numbers should match these to within a few decimal places, modulo
floating-point variation across BLAS implementations.

## Data acquisition

The audit scripts operate on publicly distributed datasets. None of the raw
EEG or rating data is shipped in this repository. Download instructions:

| Dataset | What is needed | Source |
|---|---|---|
| DEAP    | `participant_ratings.csv` (self-report SAM ratings); optionally `data_preprocessed_python/s*.dat` for the phantom-delta verification | http://www.eecs.qmul.ac.uk/mmv/datasets/deap/ (request access) |
| FACED   | `Data/sub*/After_remarks.mat` (per-subject SAM ratings) | https://www.synapse.org/#!Synapse:syn50614194 |
| SEED    | `ExtractedFeatures_1s/` (DE features, optional for the SEED audit) | https://bcmi.sjtu.edu.cn/~seed/ (request access) |

Place each dataset under `./data/` to use the default paths, e.g.:

```
data/
  faced/sub001/After_remarks.mat
  faced/sub002/After_remarks.mat
  ...
  deap/participant_ratings.csv
  deap_preprocessed_python/s01.dat
  ...
```

Each script accepts an explicit `--faced-dir` / `--deap-ratings` /
`--data-dir` flag to point elsewhere if you keep the data outside the repo.

## Repository layout

```
eeg-emotion-audit-toolkit/
  README.md                  -- this file
  LICENSE                    -- MIT
  CHECKLIST.md               -- 8-item reporting checklist
  MODELCARD_TEMPLATE.md      -- Model Card template + filled DEAP DGCNN example
  requirements.txt           -- pip dependencies
  data/
    survey_table.csv         -- 40-paper protocol survey
  scripts/
    compute_krippendorff.py      -- Section 4.3 (alpha reliability across DEAP/FACED)
    compute_cross_pipeline_r.py  -- Section 3.2 (DE cross-pipeline correlation)
    compute_noise_matrix.py      -- Section 4.4 (stim-vs-SR mismatch, MI, asymmetry)
    phantom_delta_spectrum.py    -- Appendix E (DEAP delta-band roll-off)
    verify_deap_labels.py        -- Section 4.1 (threshold sweep, ambig zone, baseline)
    verify_faced_arousal.py      -- Section 4.4 (FACED arousal mismatch + alpha)
    verify_faced_labels.py       -- Section 4.4 (FACED valence mismatch, per-category)
  results/
    cross_pipeline_r_results.json    -- reference output for compute_cross_pipeline_r.py
    deap_label_kappa_sweep.json      -- reference output for verify_deap_labels.py
```

## Script-to-paper map

| Script | Reproduces |
|---|---|
| `compute_cross_pipeline_r.py`  | Cross-pipeline DE correlation r approx 0.55, with delta band r approx 0.22 (Section 3.2) |
| `compute_krippendorff.py`      | Krippendorff's alpha values for DEAP and FACED (Table 3, Section 4.3) |
| `compute_noise_matrix.py`      | 19.8% binary stimulus-vs-self-report mismatch on FACED, mutual-information reduction, positive-vs-negative asymmetry (Section 4.4) |
| `phantom_delta_spectrum.py`    | DEAP delta-band PSD plot showing the 4 Hz high-pass cutoff (Appendix E) |
| `verify_deap_labels.py`        | DEAP ambiguous-zone fraction (~30%), seven-threshold majority-baseline sweep 56.5% to 72.2% (Sections 4.1 / 4.2) |
| `verify_faced_arousal.py`      | FACED arousal stim-vs-SR mismatch (39.5-43.1% sensitivity range), alpha for arousal (Section 4.4) |
| `verify_faced_labels.py`       | FACED valence stim-vs-SR mismatch (~19.8%), per-category mismatch rates (Section 4.4) |

## Reporting checklist (`CHECKLIST.md`)

Eight axes a paper must disclose to be reproducible:

1. Preprocessing
2. Features
3. Threshold
4. Label type
5. Split method
6. Cross-validation
7. Baselines
8. Code and data

See `CHECKLIST.md` for the full text.

## Model Card template (`MODELCARD_TEMPLATE.md`)

Three EEG-emotion-specific field groups extending the Mitchell et al. 2019 and
Kinahan et al. 2024 lineage:

- Label construction (stimulus vs. self-report, raters, alpha, threshold)
- Protocol ladder (CV regime, val/test separation)
- Leakage isolation (best-vs-worst-protocol gap, Kapoor subtype codes)

A worked example for the DEAP DGCNN controlled-leakage experiment is included.

## License

MIT (see `LICENSE`).

## Citation

Citation block to be added on acceptance. The corresponding paper is currently
under double-blind review; authorship and venue details will be filled in once
the embargo lifts.
