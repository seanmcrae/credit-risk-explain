# credit-risk-explain: product write-up

## Problem

A card issuer's collections or early-outreach team can work a few hundred to a few thousand
accounts per cycle out of a much larger book. Today that list is often built from one or two rules
("anything 2+ months late", "highest balance first"). Rules like these leave value behind in two
ways: they spend contacts on accounts that would have paid anyway, and they cannot say how many
contacts are worth making. A risk model can rank better, but an unexplained score creates a new
problem: analysts will not trust it, ops leads cannot staff against it, and compliance cannot sign
off on it.

The job is a ranking that is **better than the rule, explainable per account, calibrated enough to
reason about money, and auditable across customer groups**, delivered with the evidence each
stakeholder needs.

## Users and jobs to be done

| User | Job | What they need from this tool |
|---|---|---|
| Collections / risk analyst | "Tell me who to call first and why, so I can open the conversation well." | Ordered queue, probability, decile, top reasons in plain language, per-account waterfall |
| Ops lead | "How many agents do I need this cycle, and what do I lose if I'm short-staffed?" | Expected value by queue size, capture at a given capacity |
| Model risk / compliance | "Can I defend this model to an examiner and to a customer?" | Model card generated from the run, protected attributes excluded and audited, monotone risk drivers, reason codes that only cite risk-increasing factors |
| Data scientist (owner) | "Can I retrain, compare and promote without hand-editing reports?" | Config-driven pipeline, champion/challenger metrics side by side, deterministic runs, CI |

## Scope

**In:**
- Ranking accounts by probability of default next month from six months of statement history.
- Two models (interpretable baseline, gradient-boosted champion) with calibration.
- Queue economics under a configurable cost/benefit matrix.
- Per-account reason codes and global SHAP importance.
- Fairness slice audit and a generated model card.
- CLI for batch use, Streamlit viewer for analysts.

**Out:**
- Credit approval, limit or pricing decisions.
- Treatment selection (which script, which channel) and uplift modelling.
- Real-time scoring service, case management integration, customer contact.
- Bias mitigation (reweighting, thresholds by group): the tool measures, a human decides.

## Requirements

| # | Requirement | How it is met |
|---|---|---|
| R1 | Queue order beats a random and a linear baseline on a holdout the model never saw | Stratified train / calibration / test split; both models scored on the same test set |
| R2 | Probabilities are trustworthy enough to multiply by money | Calibrator fitted on a separate split; ECE and reliability curve on holdout |
| R3 | Every queued account carries reasons a human can read | SHAP contributions grouped into seven reason codes; only risk-increasing groups are shown |
| R4 | Risk drivers move in the defensible direction | LightGBM monotone constraints on eight features, set in config, enforced by a test |
| R5 | Protected attributes never influence the score | Excluded from features by construction; test flips sex and age and checks scores are unchanged |
| R6 | Group differences are measured and published | Priority rate, TPR/FPR at capacity, calibration gap and AUC per group, in the model card |
| R7 | Every number in the documentation comes from a run | `MODEL_CARD.md` is rendered from `metrics.json`; README figures are produced by `make train-uci` |
| R8 | Runs are reproducible and CI needs no data download or keys | Seeded split, deterministic LightGBM, synthetic sample for tests, lockfile |

## Success metrics and evals

The tool measures itself on every training run; the targets below are the bar I would set for
promoting a model in this setting. Current values are from the UCI holdout (6,000 accounts).

| Metric | Why it matters | Target | UCI run |
|---|---|---|---|
| Capture at capacity: share of defaulters in the top 20% | The number an ops lead staffs against | Beats baseline model; > 45% | 50.7% (baseline 48.2%) |
| Precision in top 5% | Analyst trust: the top of the list should mostly be real | > 70% | 77.0% |
| Expected value at configured capacity | Ties ranking quality to money under stated assumptions | Higher than working everyone and higher than baseline | 197,200 vs 170,800 (everyone) and 184,000 (baseline) |
| Calibration error (ECE, 10 bins) | Probabilities feed the economics and the analyst's judgment | < 0.02 | 0.010 |
| Score stability (PSI, development vs scoring population) | Early warning that the population moved | < 0.1 stable, 0.1-0.25 investigate, > 0.25 retrain | 0.002 (train vs holdout) |
| TPR gap across groups at capacity | Equally risky customers should be equally likely to be worked | < 0.10 per attribute, or documented and accepted | sex 0.036, marriage 0.015, education 0.085, age band 0.180 |
| Max absolute calibration gap across groups | Score means the same thing for every group | < 0.03 | 0.029 (age band 18-24) |

The age-band TPR gap fails the target: defaulters aged 18-24 are prioritized at 60.6% versus 42.6%
for 55+. In the holdout, defaulters aged 18-24 have a median credit limit of 50,000 against
100,000 for 55+, and 55% of them are currently late against 50%. The model card's proxy screen
shows how that reaches the score: among defaulters, credit limit adds 0.164 more log-odds for
18-24 than for 55+ and recent payment status 0.088 more, partly offset by average utilization
(0.173 less). Credit limit is the model's second-largest driver, so it acts as a partial proxy
for age. Whether this is acceptable depends on what being worked means for the customer, which
is why the gap is published rather than silently corrected.

Evals that run on every commit (synthetic data): hand-computed KS, lift and expected-value cases,
reason-code mapping, monotonicity of the constrained model, deterministic retraining, invariance
to protected attributes, CLI and app smoke tests.

## Minimum viable quality

Release thresholds for a retrained model, built on the targets in the table above. Do-not-ship is
a hard stop; ship means every target is met; delight is the bar for a model I would promote with
no caveats.

| Metric | Do not ship | Ship | Delight |
|---|---|---|---|
| Capture at 20% capacity | 45% or less, or not above the baseline model | Above 45% and above the baseline model | Also above a rule-based queue (not yet scored, issue #4), on an out-of-time sample |
| Precision in top 5% | 70% or less | Above 70% | Above 75% on an out-of-time sample |
| Expected value at configured capacity | Not above working everyone, or not above the baseline | Above both | Above both across a sensitivity band of cure rate and loss given default |
| Calibrated ECE (10 bins) | 0.02 or more | Below 0.02 | Below 0.01 |
| Score PSI, development vs scoring | Above 0.25 | Below 0.1 (0.1-0.25: investigate before shipping) | Below 0.1 on an out-of-time sample |
| TPR gap per attribute at capacity | 0.10 or more with no recorded acceptance by the program owner | Below 0.10, or documented and accepted | Below 0.10 for every attribute, no exceptions |
| Max absolute calibration gap across groups | 0.03 or more | Below 0.03 | Below 0.02 |

Against these thresholds the current UCI run meets ship on every row but one: the age-band TPR gap
is 0.180. It is documented, but no acceptance is recorded (issue #3), so as it stands this model is
do-not-ship until the program owner accepts the gap or a mitigation narrows it. The 18-24
calibration gap (0.029) clears the bar by 0.001.

## Cost at 1x and 10x usage

Estimates only. The repo's cost assumptions are the illustrative economics in
`configs/default.yaml`: 60 per contact, 4,000 loss given default, a 10% cure rate if worked (so
working an account pays off above a calibrated probability of 0.15). The config names no currency.
1x is the UCI holdout: a book of 6,000 accounts per cycle at the 22.1% default rate, worked at 20%
capacity. 10x assumes the same default rate and the same model performance on a book ten times the
size, which is an assumption, not a measurement.

| Per cycle | 1x (6,000 accounts) | 10x (60,000 accounts) |
|---|---|---|
| Accounts worked at 20% capacity | 1,200 | 12,000 |
| Contact cost | 72,000 | 720,000 |
| Expected value at 20%, LightGBM | 197,200 | about 1,972,000 |
| Expected value at 20%, logistic-regression baseline | 184,000 | about 1,840,000 |
| Expected value of working every account | 170,800 | about 1,708,000 |
| Value-maximizing capacity, LightGBM | 3,102 accounts (51.7%), 237,880 | about 31,000 accounts, about 2,378,800 |

Contact cost and value scale linearly because every figure is per account; what does not scale is
team capacity, which is why the queue size is a staffing decision rather than a model output.
Training and scoring run as a local batch job, and the repo has no compute cost assumption, so
compute is not costed here.

## Trade-offs and alternatives considered

- **Interpretability vs accuracy.** Logistic regression is easier to defend; LightGBM is
  materially better here (ROC-AUC 0.778 vs 0.738, capture at 20% 50.7% vs 48.2%). Monotone
  constraints plus SHAP close most of the explainability gap at no measured accuracy cost, so
  LightGBM is champion and the linear model stays as challenger and sanity check.
- **Which fairness definition.** Demographic parity (equal priority rates) would force working
  lower-risk customers in some groups and fewer high-risk ones in others. Equal opportunity (equal
  TPR) is closer to "equally risky people are treated alike". The tool reports both plus
  calibration by group and leaves the policy choice to the program owner.
- **Rank on calibrated vs raw score.** Calibrated isotonic output ties whole blocks; ranking on it
  would throw away ordering information. Raw score for order, calibrated probability for
  decisions.
- **Isotonic vs Platt.** Isotonic fits the non-linear miscalibration of boosted trees better;
  Platt is safer with small calibration sets. Both are a config switch. Isotonic levels are
  Laplace-smoothed so small extreme blocks do not report probability 1.0.
- **Stratified vs out-of-time split.** Out-of-time is the right test for a production model, but
  the public data has a single observation month. The PSI check and `evaluate --data` are the
  hooks for an out-of-time sample.
- **Expected value from realized outcomes vs predicted probabilities.** The holdout curve uses
  realized defaults (what would actually have happened); the app uses calibrated probabilities
  because a live queue has no labels yet.

## Risks

| Risk | Mitigation in this repo | Remaining exposure |
|---|---|---|
| Proxy discrimination through limit or utilization | Group audit with TPR, FPR and calibration gaps | Gaps are measured, not mitigated |
| Placeholder economics drive staffing decisions | Assumptions printed next to every value figure; all in config | Real cure rates and losses must come from the business |
| Model trained on 2005 Taiwan applied elsewhere | Stated in model card and README limitations | No transfer validation possible on public data |
| Reason codes read as causal or as legal notices | Wording and limitation text say they describe the model | Legal review needed before customer-facing use |
| Silent drift after deployment | PSI computed every run; `evaluate --data` on new labeled files | No scheduled monitoring yet (roadmap) |

## Roadmap

**Now (in this repo):** calibrated champion/challenger ranking, queue economics, grouped reason
codes, fairness audit, generated model card, CLI, Streamlit viewer, CI on Python 3.11 and 3.12.

**Next:**
- Drift monitoring: per-feature and score PSI against the development sample on every scoring
  batch, with thresholds that open a retrain ticket.
- Champion/challenger promotion: a `compare` command that scores both bundles on the same new
  labeled month and promotes only if capture at capacity improves without widening group gaps.
- Out-of-time validation as soon as a dataset with an observation date is available.
- Capacity sensitivity: expected value bands across ranges of cure rate and loss given default.

**Later:**
- Uplift modelling: rank by expected change in default from contact, not by default risk alone.
- Group-aware policy options (for example, TPR-equalizing capacity allocation) shown as explicit
  trade-off curves for the program owner to choose from.
- Reason-code stability tracking, so an account's stated reasons do not flip between cycles
  without a real change in its history.
