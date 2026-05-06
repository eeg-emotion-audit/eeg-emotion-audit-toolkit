# Papers considered but not included in the protocol survey

This file records cross-subject EEG emotion-recognition benchmark candidates that we evaluated for inclusion in the 45-record protocol survey (`survey_table.csv`) but ultimately excluded. Each entry lists the paper, the reason for exclusion, and the source/discovery context.

## Excluded — methodological / protocol caveats

| Paper | Year | Venue | Reason for exclusion |
|---|---|---|---|
| MADA (Qiu et al.) | preprint | eScholarship | Not peer-reviewed (UC repository preprint); target-peeking source-domain selection (DDSR uses small target-domain labeled subset to rank source domains pre-transfer) |
| TDMNN (Wang et al.) | 2023 | Cogn. Neurodyn. | Subject-dependent 99% mislabeled as cross-subject; LOSO not reported as primary result |
| MdGCNN-TL (Bi et al.) | 2022 | Neural Comput. Appl. | Wrong task (3-class on DEAP, not binary V/A) |
| DADPc (Dan et al.) | 2025 | Front. Neurosci. | Uses DEAP only as cross-database probe (DEAP↔SEED-session); 14-of-32 sub-selection; not within-DEAP LOSO |
| EEG-SCMM (Deng et al.) | 2024 | arXiv 2408.09186 | SEED3→DEAP-Arousal cross-corpus benchmark, not within-DEAP LOSO |
| MGFKD (She et al.) | 2023 | Comp. Biol. Med. | Quasi-transductive (target-label-aware source ranking); single-session only |
| MS-MDA (Chen et al.) | 2021 | Front. Neurosci. | Protocol mismatch with our 15-fold LOSO definition (uses 14→1 per session averaged over 3 sessions); already audited separately for val=test handling |

## Excluded — protocol-loosening pattern (2024-2025 SEED ceiling-climb)

These 2024-2025 papers report SEED LOSO accuracies above 90% but each has a documentable protocol caveat. Including them would amplify the inflation pattern we audit rather than represent the field's honest baselines.

| Paper | Year | Venue | Reported (SEED/SEED-IV) | Caveat |
|---|---|---|---|---|
| TT-CDAN (Huang et al.) | 2025 | IEEE TAFFC | 93.62 / 82.16 | Authors themselves call traditional offline LOSO inflated; propose stricter online protocol |
| PMDA-RSCG (Chen et al.) | 2025 | IEEE TNSRE | 97.03 / 88.18 | EEG-CutMix between source and target during training (likely leakage vector); >9pp above all 2018-2024 baselines |
| CPDAN (Cheng et al.) | 2025 | Cogn. Neurodyn. | 88.06 / 81.29 | Uses 13-train + 2-test subjects (leave-2-out), not LOSO 15-fold; SEED-IV result is 7-12pp above paper's own comparators |
| EEGMatch (Zhou et al.) | 2025 | IEEE TNNLS | 92.39 (SEED, N=12 labeled) / 66.29 (SEED-IV, N=20) | Semi-supervised with incomplete labels, not pure DA-with-full-labels benchmark |
| SS-EMERGE | 2025 | Sci. Reports | 92.35 / 81.51 | Self-supervised, not pure DA |
| DS-AGC (Ye et al.) | 2025 | IEEE TAFFC | 87.30 / 66.00 | Comparable tier to RGNN/BiHDM already in survey; PR-PL and DCGNN+CDAN cover the 2024-2025 DA tier |

## Excluded — task or paradigm mismatch (FACED)

| Paper | Year | Venue | FACED result | Caveat |
|---|---|---|---|---|
| FreqDGT (Li et al.) | 2025 | MIND conf. | binary 62.3 ± 8.9 | Uses self-report valence threshold 3.0 to binarize FACED — DEVIATES from Chen 2023's stimulus-category protocol; FreqDGT's CLISA reproduction = 58.6% (vs Chen's 75.1%) confirms protocol incompatibility |
| GCPL (Fan/Shao et al.) | 2024 | — | 9-class 36.87 ± 3.27 | Designed for partial-label / self-report ambiguity; using on stimulus labels is off-task |
| CL-CS | 2025 | Biomed. Signal Process. Control | binary 72.5 ± 15.3, 9-class 43.4 ± 13.7 | Variance × 2 ≈ 30 pp range (unreliable headline) |

## Excluded — secondary reproductions (not native benchmark)

| Paper | Status |
|---|---|
| DNN_AER (Sujatha et al.) | Originally speech recognition; FACED numbers come from CSCL's reproduction, not native benchmark |
| DAPLP (Zhong et al.) | Only via CSCL's reproductions; not native benchmark |
| CSMM (Zhu et al.) | Originally SEED multimodal; FACED via CSCL reproduction only |

## Honorable mentions (not verified line-by-line)

These were surfaced via web search but were not verified line-by-line in this survey window. Could be added in future revisions.

| Paper | Year | Venue | Reported (DEAP V/A) | Status |
|---|---|---|---|---|
| Quan et al. MR-VAE | 2023 | Biomed. Signal Process. Control | 81.19 / 79.59 | on disk |
| MSRN + MTL (Li et al.) | 2022 | — | 71.29 / 71.92 | on disk |
| Liu et al. multi-source | 2023 | Biomed. Signal Process. Control | DREAMER 72.84 V / 82.62 D / 82.72 A (suspiciously high) | on disk |

## Selection criterion (recap)

The included 45-record survey samples roughly the most-cited cross-subject EEG emotion-recognition benchmarks per dataset published 2018-2025 — approximately the top 10 for DEAP and FACED, the top 5 for SEED and SEED-IV, and the available cross-subject benchmarks for DREAMER. The sample intentionally spans the credibility spectrum (HIGH-MEDIUM-LOW) to surface the reporting-discipline failures the audit characterizes; the entries above were excluded because their inclusion would either (a) introduce a non-representative protocol artifact, (b) duplicate an already-covered audit theme, or (c) require additional line-by-line verification we could not complete in the survey window.
