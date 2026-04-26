# EEG-Emotion Model Card Template

This template extends the general-purpose Model Card framework of Mitchell et
al. (2019) and the EEG-machine-learning Model Card of Kinahan et al. (2024)
with three domain-specific field groups for EEG-based emotion recognition:
**label construction**, **protocol ladder**, and **leakage isolation**.

The Mitchell 2019 and Kinahan 2024 fields (intended use, evaluation context,
ethical considerations, channel-selection rationale, preprocessing pipeline)
are retained from the parent frameworks and are not reproduced here in full;
a complete card would populate those as well. The design goal is not to
replace existing Model Card work but to specialise it for the label-noise,
protocol-ladder, and leakage-subtype vocabulary required by EEG-emotion audits.

Fields that cannot be populated from the published material of a given paper
should be marked explicitly (e.g., "not reported") rather than omitted, since
missingness itself carries reproducibility signal.

---

## Three EEG-emotion-specific field groups

### Label construction

| Field | Expected value |
|---|---|
| Stimulus vs. self-report | Categorical tag: `stimulus-based`, `self-report`, or `hybrid`. Hybrid schemes (e.g., stimulus with SAM confirmation) should be flagged. |
| Number of raters | For self-report, the number of subjects each rating their own trials; for stimulus-based labels, the number of independent annotators. |
| Krippendorff's alpha | Inter-rater reliability at the binarization threshold in use. Flag alpha < 0.67 as below the conventional reliability threshold. |
| Binarization threshold | Numerical cut applied to the continuous rating (e.g., 5.0 on a 1-9 scale, per-subject median, or fixed bin boundaries). The corresponding majority-class baseline should be reported alongside. |

### Protocol ladder

| Field | Expected value |
|---|---|
| CV regime | One of K-Fold / LNSO / LOSO / N-LNSO / N-LOSO, following the five-rung ladder of Del Pup et al. The rung should be reported together with the subject-pooling convention (disjoint subjects vs. trial splits within subjects). |
| Validation protocol | Whether a held-out validation set is used for early stopping and hyperparameter selection, or whether the test set is reused for these purposes ("val = test"). Disclosure of the validation split convention is necessary for val = test detection. |

### Leakage isolation

| Field | Expected value |
|---|---|
| Best- vs. worst-protocol gap | Difference in reported accuracy between the most favourable and most honest evaluation protocol applied to the same data. Where only one protocol is run, a lower-bound estimate from literature (e.g., the ~+37 pp within-trial figure for DEAP valence) should be cited. |
| Kapoor subtype code(s) | Which of L1.1, L1.2, L1.3, L1.4, L2, L3.1, L3.2, L3.3 (Kapoor and Narayanan 2023) the reported numbers are demonstrably free of, and which remain a residual risk. Silence on a subtype is treated as unresolved rather than absent. |

---

## Filled instance: DEAP DGCNN controlled-leakage experiment

A worked example, populated for the controlled leakage experiment described
in Section 5.1 of the paper. Values refer to the clean, trial-level-split
condition.

### Label construction

| Field | Value |
|---|---|
| Stimulus vs. self-report | self-report (binary valence from the 1-9 SAM rating) |
| Number of raters | 32 subjects, each rating their own trials |
| Krippendorff's alpha | 0.41 to 0.48 across binarization thresholds 4.5 to 5.5; below the 0.67 conventional reliability threshold |
| Binarization threshold | 5.0 on the 1-9 scale; majority-class baseline 56.5% |

### Protocol ladder

| Field | Value |
|---|---|
| CV regime | Subject-dependent within-subject 10-fold for the controlled leakage experiment; LOSO 32-fold for the cross-subject ceiling number. Both use trial-disjoint fold assignment. |
| Validation protocol | val != test; early stopping uses a separate held-out validation fold disjoint from the test fold. |

### Leakage isolation

| Field | Value |
|---|---|
| Best- vs. worst-protocol gap | +36.72 pp between trial-level and within-trial window-level splitting, with DGCNN backbone, DE features, and seed fixed. |
| Kapoor subtype code(s) | Reported numbers are free of L1.1 (held-out test exists), L1.2 (preprocessing fitted on training data only), L2 (no label-derived features), L3.1 (within-trial temporal leakage controlled by trial-level split), and L3.2 (subject-disjoint LOSO for the 32-fold numbers). L1.3, L1.4, and L3.3 are not applicable to DEAP binary valence. |

---

## Deriving a card from an existing paper

A full card for any single paper in the field can be derived from the paper
text, source code (where released), and the released results, by populating
the three field groups above. Where no code or data release is available, a
best-effort card based on the paper text alone is still informative, provided
that unpopulated fields are marked explicitly.

## References

- Mitchell, M., et al. (2019). Model cards for model reporting. *FAccT*.
- Kinahan, S., et al. (2024). Model cards for EEG machine learning. *FAccT*.
- Kapoor, S., and Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science.
- Del Pup, F., et al. (2025). eegpartition: nested-CV partitioning library for EEG.
- Krippendorff, K. (2011). Computing Krippendorff's alpha-reliability.
