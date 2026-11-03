# Research integrity audit — 3 November 2026

The original archive was preserved outside this extracted project. Treat its PDF, TeX, figures and JSON as inherited, unaudited artifacts. The public release omits the old paper and saved v0.2 tables so they cannot be mistaken for newly validated findings.

| Finding | Correction / consequence |
| --- | --- |
| Leap used nested significant subsets and invented a full-support endpoint | Implements closure under union of prior monomial supports. Missing support after coefficient thresholding returns unresolved (`None`). Regression tests include a non-nested staircase. |
| Fourier helper described as learning from data | Explicitly documented as target-oracle access using counterfactual labels and support metadata. Ordinary dataset-only prediction remains future work. |
| PAC-Bayes chose prior variance on the training set without a union correction | Allocates delta across a prespecified finite prior grid; also upper-bounds Monte Carlo empirical risk via Hoeffding over independent posterior weight draws. |
| Non-vacuous was conflated with error below 0.5 | Uses <1 for non-vacuous, separate `beats_chance` field. The certificate concerns a randomized Gibbs classifier. Old numeric certificates must be recomputed. |
| Predictor replaced Rademacher correlation with half the generalization gap | Legacy script now measures random-label fitting correlation; old predictor coefficients/correlations are superseded and are not used in the app. |
| Unobserved crossings were replaced by twice the largest tested n | New benchmark preserves interval/left/right censoring. Legacy regression path now rejects right-censored observations rather than silently imputing them. |
| Fixed epochs confounded sample size with optimizer updates | New benchmark fixes SGD update count across n and records achieved updates and elapsed time. |
| One initialization seed for most old claims | Data, hidden support and weight initialization vary across three seeds. Still too few seeds to support a publication claim. |
| Post-hoc LI called a pre-training predictor; exponent claimed inside LI | LI described as exploratory post-hoc score; exponent is separate. Ablation and held-out predictive evaluation remain necessary. |
| Truncated Gaussian claimed dimension-free bounded-radius margin guarantee | Metadata explicitly notes unbounded input norms; that theorem does not directly apply. |
| Fixed-target parity VC dimension 1 used to describe unknown support | Removed misleading value. Statistical baseline actually solves an unknown parity using only observed equations. |
| Linear learners said to never beat chance on parity | No such blanket guarantee: k=1 is linear and higher-order threshold rules can obtain partial accuracy. Finite failures are configuration-specific. |
| Merged-staircase citation dated 2023 | Distinguishes COLT 2022 merged-staircase paper and COLT 2023 leap paper. |
| App allowed k>d; notebook pretended to be self-contained | App bounds k by d; notebook requires package or safe ZIP extraction and executes the real benchmark. |

Remaining limits: noiseless synthetic data, sparse grid, 1,000 test observations per fit, pointwise intervals, three seeds, no new theorem, no verified novelty, and no prospective held-out predictor claim. In the fixed-update experiment, duplicate inputs are possible because samples are i.i.d. on a finite Boolean cube; train/test draws are independent rather than guaranteed distinct.

A publishable paper needs a revised research question, independent related-work assessment, more target/support and initialization seeds, optimization-budget sweeps, held-out families/dimensions, baseline and feature ablations, censoring-aware evaluation and uncertainty across training runs. The inherited PDF should not be submitted or called a publication.
