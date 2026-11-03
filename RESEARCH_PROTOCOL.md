# Proposed study: budget-conditioned learnability forecasting

Status: proposed extension, not an established novel contribution or completed study.

## Question
Can low-cost diagnostics from an ordinary labeled probe set forecast which (data size, optimization budget) configurations will learn a held-out target, beyond majority-label, dimension-only, class-frequency and short-training baselines?

## Design to freeze before evaluating

1. Use at least ten support seeds and ten initialization seeds per target family. Vary parity, non-nested staircases, decision lists and norm-controlled halfspaces; do not treat initialization repeats as independent target functions.
2. Cross dimensions 12, 24, 48 and 96 with widths and optimizer budgets. Report fixed-update, fixed-epoch and matched-wall-time controls separately. Do not infer a universal exponent from a narrow grid.
3. Separate training data, a diagnostic probe set, threshold-selection validation, and an untouched final test draw. Probe data and oracle calls count toward the diagnostic cost. Keep oracle Fourier features in an explicitly privileged ablation.
4. Compare centered and uncentered NTK alignment, label entropy, spectral diagnostics, a small-sample learning curve and a cost-matched warm-up baseline. Fit simple dimension-only and constant predictors before adding features.
5. Split complete target instances and dimensions into train/validation/test groups. Report within-family and leave-family-out transfer separately. Do not pool correlated repeats into random row splits.
6. Record target-error thresholds, all failures and runtime limits before running. Preserve left/right/interval censoring; evaluate predicted probability of success at each budget instead of inventing unobserved exact transitions.
7. Use group bootstrap intervals over target instances, calibrated success probabilities, Brier score, held-out likelihood and coverage. Correct for multiple comparisons where inferential claims require it. Release all configs and failed runs.

## Why the 2025/2026 literature is relevant

Lyu et al., *Transformers with RL or SFT Provably Learn Sparse Boolean Functions, But Differently*, first posted 22 November 2025 and revised for ICML 2026, studies one-layer transformers learning decomposable sparse Boolean tasks with intermediate reasoning supervision. It motivates a future supervision-access axis for the benchmark. Adding a transformer is a separate implementation and reproduction task; it is not achieved by renaming the current MLP experiments.

Source: https://arxiv.org/abs/2511.17852
Foundations: https://proceedings.mlr.press/v178/abbe22a.html and https://proceedings.mlr.press/v195/abbe23a.html

## Decision criteria

Proceed toward a workshop submission only if a prespecified method improves held-out prediction or reveals a robust counterexample across seeds and budgets. A careful negative result can be useful, but the contribution must be distinguished from existing learning-curve forecasting, kernel alignment and sparse-function SGD literature. Seek a learning-theory researcher to review the study before claiming novelty.

## Application wording

Suggested project description (adapt after personally reviewing the code and reproducing the experiments):

“Developed a CPU-based learning-theory workbench with a NumPy SGD implementation, an exact GF2 parity baseline, and reproducible budget-controlled benchmarks. Evaluated 108 model fits with independent validation/test draws, error intervals, censored transitions, and configuration/source provenance; added an interactive demo and automated tests.”

Use “research project” or “empirical study.” Do not say “published paper,” “proved computational hardness,” or “universal pre-training learnability predictor.” Admissions value will depend on your understanding, independent contribution, experimental depth and the rest of your application; no project guarantees admission.
