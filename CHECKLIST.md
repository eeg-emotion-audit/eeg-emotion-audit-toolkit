# Reporting Checklist for EEG Emotion Recognition

A minimum reporting checklist to make EEG-based emotion-recognition results
reproducible and comparable across papers. Each item below corresponds to
one of the audit dimensions documented in the accompanying paper. Items not
applicable to a given study should be marked explicitly rather than omitted;
silence on an item is treated as an unresolved risk, not as the absence of one.

## 1. Preprocessing

Specify the bandpass range, reference scheme, artifact-rejection method, and
whether the preprocessed release or raw recordings were used. If raw recordings
are re-preprocessed, disclose the pipeline at a level of detail that allows
re-running it (filter type/order, cutoff frequencies, downsampling, channel
selection, artifact-rejection algorithm and parameters).

## 2. Features

Report the feature type (DE, PSD, raw EEG, learned representation), the
frequency-band definitions, and the segment length. Verify that all reported
frequency bands fall within the data bandwidth: e.g., the DEAP preprocessed
release applies a 4-45 Hz bandpass, so any "delta-band" feature extracted from
this release contains only filter roll-off, not neural signal.

## 3. Threshold

State the binarization threshold used to convert continuous valence or arousal
ratings to class labels, and report the corresponding majority-class baseline.
Different thresholds shift the majority-class baseline by 16 percentage points
on DEAP (56.5% to 72.2%); a reported accuracy is uninterpretable without its
threshold and baseline.

## 4. Label type

Specify whether stimulus-based or self-report labels are used. For self-report
labels, report Krippendorff's alpha (or equivalent inter-rater reliability)
at the binarization threshold in use. Flag alpha < 0.67 as below the conventional
reliability threshold; on DEAP and FACED self-report binary valence, alpha
typically falls in the 0.41-0.48 range.

## 5. Split method

Describe whether train/test splits are at the trial level or the window level.
Report the number of segments per trial. Within-trial splitting (adjacent
windows from the same recording in train and test) inflates subject-dependent
accuracy by ~+37 percentage points relative to trial-level splitting; the
distinction between trial-level and window-level splits is the single most
important methodological detail in protocol reporting.

## 6. Cross-validation

Specify the fold-assignment scheme (sequential, shuffled, random seed) and
whether subject pooling is disjoint at the subject level (LOSO, LNSO) or at
the trial level within subjects (subject-dependent K-fold). For LOSO, state
explicitly whether a separate validation set is used for early stopping and
hyperparameter selection, or whether test performance is reported directly
(the "val = test" pathway, which inflates reported accuracy via implicit
test-set model selection).

## 7. Baselines

Report the majority-class baseline. Where computationally feasible, also
report a shuffled-label control: a model trained on randomly shuffled labels
should achieve approximately chance accuracy. Numbers far above chance under
shuffled labels indicate residual leakage that the trial/window split alone
does not control.

## 8. Code and data

Release preprocessing and evaluation code in a public repository. State the
exact dataset version used, the random seed for any non-deterministic
component, and the specific dataset partition (subjects, trials, sessions)
each reported number was computed on. For results trained on a derived
preprocessed release, indicate which preprocessing pipeline was used and
whether it matches the official release.

## Recommended reference implementations

Three concurrent library efforts converge on an equivalent three-way
train / validation / test split prescription that precludes the val = test
pathway by construction:

- **LibEER** -- 6:2:2 train/validation/test split with val-best model
  selection.
- **LibEMER** -- 3:1:1 train/validation/test split.
- **eegpartition (Del Pup et al.)** -- the
  `create_nested_kfold_subject_split` function returns disjoint
  training / validation / test triples at the subject level.

Adoption of any of these, or of an equivalent protocol that enforces the
three-set separation, is recommended as the reference implementation for
cross-subject EEG emotion recognition.

## Companion artifacts

This checklist is complemented by a domain-specific Model Card template
(`MODELCARD_TEMPLATE.md`) extending the Mitchell et al. 2019 / Kinahan et
al. 2024 lineage with three additional field groups for label construction,
protocol ladder, and leakage isolation.
