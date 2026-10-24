# FAITHDROID

Automatically measurable explanation faithfulness for cost-aware LLM-based Android malware
detection under concept drift.

MSc research project (Cybersecurity) — Vinay, National College of Ireland.

---

## About this package

This is the code submission, packaged separately from the report and configuration manual
(submitted as their own PDFs). It contains everything needed to rerun the pipeline: `code/`,
the processed data splits and LLM response cache under `data/`, saved experiment output under
`results/`, and generated `images/`. The report and configuration manual LaTeX sources are not
included here.

Every step in **Running the pipeline** and **Seeing it work on one application** below runs
standalone from this package. Three steps write into or read from a `report/` directory that
is part of the separately-submitted report source, so they are not runnable from this package
alone: `make_tables.py` and `make_macros.py` (regenerate the report's LaTeX tables/macros) and
`verify_report.py` (cross-checks the report's prose against `results/`). `test_pipeline.py`
(the 34 invariant tests) has no such dependency and runs fully standalone.

---

## What this is

Prior LLM-based Android malware detectors claim explainability but never measure it, and
spend LLM capacity on every input. TraceRAG costs \$6.00 per application and validated its
explanations with a 24-application expert study in which reviewers confirmed only 53.33% of
verdicts; LAMD costs \$0.199 per application, paid on 100% of inputs, and never evaluates its
explanations at all.

FAITHDROID is a four-stage cascade that addresses both problems:

1. **Hybrid evidence base** — 288 system-call (dynamic) + 166 permission (static) features
   from KronoDroid, with timestamps enabling genuine concept-drift evaluation.
2. **Cheap detector + conformal calibration** — XGBoost, wrapped in a Mondrian inductive
   conformal predictor.
3. **Uncertainty-gated escalation to a local LLM** — only the ambiguous minority reaches the
   reasoner, which receives a *closed evidence bundle* plus labelled neighbours retrieved
   from the training period, and must cite artifacts only from that closed set.
4. **Automatic faithfulness audit** — Evidence Grounding Rate, Counterfactual Necessity and
   Attribution Agreement, none of which requires human annotation.

The closed citation vocabulary is the key idea: because the permitted vocabulary is finite
and known, every citation is machine-checkable, so hallucination becomes measurable at
arbitrary scale.

Everything runs locally on open-weight models. **No API cost.**

---

## Requirements

- Python 3.11+ (developed on 3.13)
- `pandas numpy scikit-learn xgboost shap matplotlib pyarrow`
- [Ollama](https://ollama.com) with the models pulled:
  ```bash
  ollama pull qwen2.5:7b
  ollama pull llama3.2:3b
  ollama pull phi4:14b
  ```
- TeX Live / TinyTeX for the report (`pdflatex`, `bibtex`, `booktabs`, `harvard`)

Developed on an Apple M5 Pro (24 GB). Any machine that can serve a 7B model will do; the
reasoning stage is the only slow part.

---

## Getting the data

KronoDroid is public and needs no API key:

```bash
cd data/raw && git clone --depth 1 https://github.com/aleguma/kronodroid.git
cd kronodroid && for f in real_device/*.zip; do unzip -o -q "$f" -d extracted/; done
```

Only the **real-device** CSVs are used. The emulator malware CSV has 483 columns rather than
484 (it lacks `EarliestModDate`, `HighestModDate`, `FilesInsideAPK`), so mixing the two would
silently introduce missing values in exactly the timestamp fields this study depends on.

---

## Running the pipeline

Run in order from the `code/` directory.

```bash
pip install -r requirements.txt
cd code

python3 preprocess.py           # build splits           -> data/processed/
python3 baselines.py            # 4 learned baselines    -> results/baselines.json
python3 conformal.py            # conformal calibration  -> results/conformal_pvalues.json
python3 routing_diagnostic.py   # routing signal choice  -> results/routing_diagnostic.json
python3 run_experiments.py      # all LLM configurations -> results/llm_*.json
python3 run_ceiling.py          # LLM-only reference     -> results/llm_ceiling_*.json
python3 faithfulness.py         # EGR / CFN / AA         -> results/faithfulness.json
python3 cascade.py              # assemble + cost curve  -> results/cascade.json
python3 statistics.py           # bootstrap CIs, McNemar -> results/statistics.json
python3 family_analysis.py      # per-family breakdown   -> results/family_analysis.json
python3 figures.py              # all figures            -> images/
python3 make_tables.py          # LaTeX tables           -> report/text/generated_tables.tex
python3 verify_report.py        # assert prose == results
python3 test_pipeline.py        # 24 invariant tests
```

`conformal.py` must run before `run_experiments.py`, because the escalation ranking is read
from its saved scores.

## Regenerating the LLM cache

The response cache key is a hash of the prompt, and the prompt depends on which samples the
detector escalates and which neighbours retrieval returns. Anything that moves either
invalidates the affected entries.

> **Retrieval tie-breaking changed (2026-08-05).** `NeighbourIndex.query` now selects
> neighbours with a stable sort instead of `np.argpartition`, whose ordering among
> equally-similar candidates is unspecified and varied between numpy builds. This makes the
> cache key reproducible, but **invalidates the 2,465 `context`-mode cached responses**; the
> 419 `grounded` and `freeform` ablation entries carry no neighbours and stay valid.

Regeneration needs a running Ollama server, then the whole downstream chain:

```bash
python3 code/run_experiments.py --ceiling   # regenerates llm_*.json
python3 code/adversarial.py --split Test1
python3 code/adversarial.py --split Test3
python3 code/faithfulness.py
python3 code/cascade.py
python3 code/statistics.py
python3 code/family_analysis.py
python3 code/figures.py
python3 code/make_tables.py
python3 code/make_macros.py                 # prose numbers
python3 code/verify_report.py
python3 code/test_pipeline.py
```

**The prose does not need retyping.** Numbers quoted in sentences come from
`report/text/generated_macros.tex`, which `make_macros.py` builds from the saved results
and the report inputs before any chapter — so `\GainTestOne{}` renders whatever
`statistics.json` currently says. `build.sh` regenerates both the tables and the macros, so
an ordinary build cannot show a stale figure.

`verify_report.py` therefore checks consistency rather than a duplicated copy of every
claim: that no prose macro is undefined, that `generated_macros.tex` is not stale, that no
`\ref` dangles, and that the thesis's **directional** claims still hold — Test2 remains
non-significant, the LLM-only ceiling still does not beat the detector, grounding is still
no lower when the system is fooled. Those are the assertions a rerun could genuinely
falsify. Run `verify_report.py --values` to additionally assert the specific figures shipped
with this repository.

The qualitative findings in `docs/02_findings_log.md` were re-measured on a second library
stack and held; the specific figures will move.

## Seeing it work on one application

```bash
python3 code/demo.py --split Test1
```

Runs a single app through every stage and prints what actually happened: the detector's
probability, whether the gate escalated, the retrieved neighbours, the LLM's verdict and
citations, which citations verify against the app, the counterfactual shift, and the final
decision with the reason. Useful for a live demonstration; takes a few seconds against a
warm cache.

```bash
python3 code/demo.py --find-disagreement   # a case where the two stages disagree
python3 code/demo.py --index 42            # a specific application
```

## Checking the work

Two guards run over the finished artefacts:

- `verify_report.py` asserts every hand-written number in the report prose against the saved
  result JSON, and checks for dangling `\ref`s and unknown citation keys. It exits non-zero if
  the narrative has drifted from the experiments. **112 checks.**
- `test_pipeline.py` tests the invariants whose violation would corrupt results silently:
  no train/test hash overlap, calibration strictly preceding every test window, prevalence
  pinned, no VirusTotal-derived column used as a feature, every citable artifact genuinely
  present in the app, retrieval drawing only from the training period, the vectorised
  Jaccard matching the naive computation, every split rebuilding identically from the raw
  CSVs, ranking ties breaking stably, and the override band having a single source of
  truth. **34 tests.**
- `envcheck.py` compares the installed stack against `requirements.txt` and exits non-zero
  on any mismatch.

### Reproducibility depends on the pinned stack — check it first

```bash
python3 code/envcheck.py     # exits 1 if the environment differs
```

The pipeline is deterministic *given a fixed library stack*, and not otherwise. The splits
rebuild byte-identically from the raw CSVs on any stack, but the **fitted models do not**.
Measured on scikit-learn 1.9.0 / pandas 3.0.2 / Python 3.14 against the pinned
scikit-learn 1.5.1 / pandas 2.3.3:

| model | Test1 | Test2 | Test3 |
|---|---|---|---|
| drebin_svm | +0.00 | +0.00 | −0.17 |
| deepdrebin | +6.43 | +1.70 | −2.60 |
| random_forest | +2.49 | +0.16 | +9.06 |
| **xgboost** (the Stage-2 detector) | **+2.50** | **−1.67** | **+3.56** |

Because the detector's probabilities decide which samples escalate, the escalated set moves
too, and the cached LLM responses stop matching — measured hit rate falls to 25–44%, so the
LLM stage then requires a running Ollama server. The fitted override band also moves from
`[0.3,0.7)` to `[0.5,0.7)`, because band `[0.3,0.5)` holds only 27 calibration samples and
sits within 0.005 of the reliability threshold.

None of this is caught by the invariant tests — they all still pass, because the invariants
genuinely still hold. **`results/` must come from a single stack; never mix.**

Individual LLM configurations can be run directly:

```bash
python3 llm_stage.py --split Test3 --budget 0.10 --mode context
python3 llm_stage.py --split Test3 --budget 0.10 --mode grounded    # ablation: no retrieval
python3 llm_stage.py --split Test3 --budget 0.10 --mode freeform    # ablation: no grounding
python3 llm_stage.py --split Test3 --budget 0.10 --no-cap           # ablation: no length cap
```

LLM responses are cached in `data/llm_cache/`, keyed by model, prompt mode, citation-cap
setting and a hash of the prompt. Reruns are therefore free, and an interrupted sweep resumes
without repeating work. Deleting the cache forces a full recomputation.

## Building the report

Not applicable to this package — the report and its build script are submitted separately as
their own PDF/source.

---

## Layout

```
code/          pipeline, one module per stage
  config.py            all paths, splits, hyper-parameters (single source of truth)
  envcheck.py          asserts the installed stack matches requirements.txt
  common.py            data loading, LAMD-format metrics
  preprocess.py        cleaning + prevalence-controlled temporal splits
  baselines.py         Drebin-style, DeepDrebin-style, RF, XGBoost
  conformal.py         Mondrian inductive conformal predictor
  routing_diagnostic.py  head-to-head comparison of 8 routing signals
  retrieval.py         Jaccard nearest labelled neighbours (vectorised)
  llm_stage.py         closed evidence bundle + constrained LLM reasoning
  run_experiments.py   all 10 LLM configurations
  run_ceiling.py       LLM-only reference point (100% escalation, random sample)
  faithfulness.py      EGR / CFN / AA + abstention curves
  cascade.py           decision rules, override band, cost-accuracy Pareto
  statistics.py        bootstrap CIs, McNemar tests, Wilson intervals
  family_analysis.py   per-family recall and escalation coverage
  figures.py           all report figures
  make_tables.py       LaTeX tables generated from results
  make_macros.py       prose numbers generated from results
  verify_report.py     asserts report prose against saved results
  test_pipeline.py     invariant tests
  demo.py              single-application end-to-end walkthrough
data/          processed splits, LLM response cache (raw KronoDroid not included — see
               "Getting the data" above to fetch it)
images/        generated figures
results/       all experiment output as JSON
```

The report, configuration manual (LaTeX source + PDF), and `docs/` design notes are submitted
separately and are not part of this package.

## Notes on two design decisions

**Prevalence control.** KronoDroid's per-year class balance swings from 88% benign (2011) to
~96% malware (2012). A naive temporal split would therefore confound concept drift with a
change in class prior, so every window is pinned to 10% malware — matching LAMD's 9.42–10.01%
so the operating regime is comparable.

**Routing signal.** The gate was originally built on conformal novelty and measured to be
*anti-selective* under drift (lift 0.5–0.8, at or below chance), because 36–45% of test
samples fall in the empty-prediction-set region and the signal saturates. Probability-margin
uncertainty achieves roughly 13×. The conformal predictor is retained for validity-backed
abstention, where it is near-exact. See `docs/02_findings_log.md`.
