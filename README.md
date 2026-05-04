# EEG-Emotion Benchmark Audit Toolkit

Companion code and data for an audit of EEG-based emotion-recognition
benchmarks across three independent layers of methodological variation:
**data preprocessing**, **label construction**, and **evaluation protocol**.

The toolkit packages:

- **A 35-record, 15-dimension protocol survey** (`data/survey_table.csv`)
  spanning DEAP, FACED, SEED, SEED-IV, and DREAMER.
- **Ten self-contained audit scripts** (`scripts/`) that reproduce the
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

Each script reads dataset paths from environment variables, with defaults
under `./data/`. To run with default paths, place the datasets under
`./data/` (see layout below). To use data stored elsewhere, set the
relevant environment variable for that script.

```bash
# Example: reproduce Table 4 (Krippendorff's alpha)
FACED_DIR=/path/to/faced/Data \
DEAP_RATINGS=/path/to/deap/participant_ratings.csv \
    python scripts/compute_krippendorff.py

# Example: reproduce Section 4.4 (noise transition matrix on FACED)
FACED_DIR=/path/to/faced/Data python scripts/compute_noise_matrix.py

# Example: reproduce Section 4.1 (DEAP threshold sensitivity)
DEAP_RATINGS=/path/to/deap/participant_ratings.csv \
    python scripts/verify_deap_labels.py

# Example: reproduce Appendix I (phantom delta-band PSD)
DEAP_DIR=/path/to/deap/data_preprocessed_python \
OUTPUT_DIR=./output/figures \
    python scripts/phantom_delta_spectrum.py
```

Equivalently, edit the path constants at the top of each script (each
constant is wrapped in `os.environ.get('VAR', default)`, so a direct edit
to the default also works).

Reference outputs from the paper's runs are committed in `results/`;
reproducing the cited numbers should match these to within a few decimal
places, modulo floating-point variation across BLAS implementations.

## Data acquisition

The audit scripts operate on publicly distributed datasets. None of the raw
EEG or rating data is shipped in this repository. Download instructions:

| Dataset | What is needed | Source |
|---|---|---|
| DEAP    | `participant_ratings.csv` (self-report SAM ratings); optionally `data_preprocessed_python/s*.dat` for the phantom-delta verification | http://www.eecs.qmul.ac.uk/mmv/datasets/deap/ (request access) |
| FACED   | `Data/sub*/After_remarks.mat` (per-subject SAM ratings) | https://www.synapse.org/#!Synapse:syn50614194 |
| SEED    | `ExtractedFeatures_1s/` (DE features, optional for the SEED audit) | https://bcmi.sjtu.edu.cn/~seed/ (request access) |
| SEED-IV | (survey-only; no scripts depend on it) | https://bcmi.sjtu.edu.cn/~seed/seed-iv.html (request access) |
| DREAMER | `DREAMER.mat` for the per-subject SAM verification | https://zenodo.org/record/546113 (request access) |

Place each dataset under `./data/` to use the default paths, e.g.:

```
data/
  faced/sub001/After_remarks.mat
  faced/sub002/After_remarks.mat
  ...
  deap/participant_ratings.csv
  deap/data_preprocessed_python/s01.dat
  ...
  dreamer/DREAMER.mat
```

Alternatively, point any script at data stored outside the repo by setting
the corresponding environment variable (`FACED_DIR`, `DEAP_RATINGS`,
`DEAP_DIR`, `DREAMER_MAT`, `CHEN_DE_DIR`, `DAEST_DE_DIR`, `OUTPUT_DIR`).

## Repository layout

```
eeg-emotion-audit-toolkit/
  README.md                  -- this file
  LICENSE                    -- MIT
  CHECKLIST.md               -- 8-item reporting checklist
  MODELCARD_TEMPLATE.md      -- Model Card template + filled DEAP DGCNN example
  requirements.txt           -- pip dependencies
  data/
    survey_table.csv         -- 35-record protocol survey
  scripts/
    compute_krippendorff.py        -- Section 4.3 (alpha reliability across DEAP/FACED)
    compute_cross_pipeline_r.py    -- Section 3 (DE cross-pipeline correlation)
    compute_noise_matrix.py        -- Section 4.4 (stim-vs-SR mismatch, MI, asymmetry)
    phantom_delta_spectrum.py      -- Appendix I (DEAP delta-band roll-off, 4 subjects)
    phantom_delta_spectrum_all32.py -- Appendix I (DEAP delta-band roll-off, all 32 subjects)
    verify_deap_labels.py          -- Section 4.1 (threshold sweep, ambig zone, baseline)
    verify_deap_arousal.py         -- Section 4.4 (DEAP arousal mismatch + alpha)
    verify_dreamer_arousal.py      -- DREAMER per-subject SAM verification
    verify_faced_arousal.py        -- Section 4.4 (FACED arousal mismatch + alpha)
    verify_faced_labels.py         -- Section 4.4 (FACED valence mismatch, per-category)
  results/
    cross_pipeline_r_results.json    -- reference output for compute_cross_pipeline_r.py
    deap_label_kappa_sweep.json      -- reference output for verify_deap_labels.py
```

## Script-to-paper map

| Script | Reproduces |
|---|---|
| `compute_cross_pipeline_r.py`       | Cross-pipeline DE correlation r approx 0.55, with delta band r approx 0.22 (Section 3) |
| `compute_krippendorff.py`           | Krippendorff's alpha values for DEAP and FACED (Table 4, Section 4.3) |
| `compute_noise_matrix.py`           | 19.8% binary stimulus-vs-self-report mismatch on FACED, mutual-information reduction, positive-vs-negative asymmetry (Section 4.4) |
| `phantom_delta_spectrum.py`         | DEAP delta-band PSD plot for 4 representative subjects showing the 4 Hz high-pass cutoff (Appendix I) |
| `phantom_delta_spectrum_all32.py`   | DEAP delta-band PSD across all 32 subjects (Appendix I, full evidence) |
| `verify_deap_labels.py`             | DEAP ambiguous-zone fraction (~30%), seven-threshold majority-baseline sweep 54.1% to 72.2% (Sections 4.1 / 4.2) |
| `verify_deap_arousal.py`            | DEAP arousal stim-vs-SR mismatch and alpha (Section 4.4) |
| `verify_dreamer_arousal.py`         | DREAMER per-subject SAM rating verification |
| `verify_faced_arousal.py`           | FACED arousal stim-vs-SR mismatch (39.5-43.1% sensitivity range), alpha for arousal (Section 4.4) |
| `verify_faced_labels.py`            | FACED valence stim-vs-SR mismatch (~19.8%), per-category mismatch rates (Section 4.4) |

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

This repository is dual-licensed:

- **Code** (`scripts/`, `requirements.txt`): **MIT License** — see `LICENSE`.
- **Data, reference outputs, and documentation** (`data/`, `results/`,
  `README.md`, `CHECKLIST.md`, `MODELCARD_TEMPLATE.md`): **Creative Commons
  Attribution 4.0 International (CC BY 4.0)** — see `LICENSE-DATA.md`.

## Citation

Citation block to be added on acceptance. The corresponding paper is currently
under double-blind review; authorship and venue details will be filled in once
the embargo lifts.
