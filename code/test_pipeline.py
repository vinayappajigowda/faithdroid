"""Invariant tests for the pipeline.

These do not test that the code runs -- the experiments already demonstrate that. They
test the properties whose violation would silently corrupt the results while everything
still appeared to work. Each one corresponds to a mistake that was actually possible, and
two of them correspond to mistakes that actually occurred during development.

Run:  python3 code/test_pipeline.py
"""
from __future__ import annotations

import json
import re
import sys

import numpy as np
import pandas as pd

import common as K
import config as C

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(f"{name}{(' - ' + detail) if detail else ''}")
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")


# --------------------------------------------------------------------- splits
def test_no_temporal_leakage() -> None:
    """No application may appear in both a training and a test split.

    A duplicate across the boundary would let the model memorise a test sample and
    inflate every downstream number, while leaving all diagnostics looking healthy.
    """
    hashes = {s: set(K.load(s)["sha256"]) for s in K.SPLITS}
    train = hashes["train"] | hashes["calib"]
    for s in K.TEST_SPLITS:
        overlap = train & hashes[s]
        check(f"no train/{s} hash overlap", not overlap,
              f"{len(overlap)} shared" if overlap else "")


def test_calibration_precedes_test() -> None:
    """Calibration must be strictly earlier than every test window."""
    cal_max = K.load("calib")["ts"].max()
    for s in K.TEST_SPLITS:
        t_min = K.load(s)["ts"].min()
        check(f"calib ends before {s} begins", cal_max < t_min,
              f"calib max {cal_max.date()} vs {s} min {t_min.date()}")


def test_prevalence_pinned() -> None:
    """Every split must sit at the target prevalence, or drift is confounded with prior
    shift. A chronological calibration split silently broke this once already."""
    for s in K.SPLITS:
        p = float(K.load(s)["Malware"].mean())
        check(f"{s} prevalence ~= {C.TARGET_PREVALENCE}",
              abs(p - C.TARGET_PREVALENCE) < 0.005, f"got {p:.4f}")


def test_no_label_leakage_in_features() -> None:
    """VirusTotal-derived columns encode the label and must never be features."""
    banned = {"Detection_Ratio", "Scanners", "TimesSubmitted", "Malware"}
    leaked = banned & set(K.FEATURES)
    check("no VT-derived column used as a feature", not leaked, str(leaked))


# ---------------------------------------------------------------- evidence bundle
def test_bundle_is_closed_and_true() -> None:
    """Every artifact offered for citation must actually be present in the app.

    This is the precondition for EGR meaning anything: if the bundle listed an artifact
    the app lacks, a "grounded" citation would not imply the app has it.
    """
    from llm_stage import build_bundle
    df = K.load("Test3")
    bad = 0
    for i in range(0, 300, 7):
        row = df.iloc[i]
        _, allowed = build_bundle(row)
        for art in allowed:
            if float(row.get(art, 0)) <= 0:
                bad += 1
    check("every bundle artifact is present in the app", bad == 0,
          f"{bad} violations")


def test_bundle_excludes_unnamed_syscalls() -> None:
    """SYS_nnn columns are classifier features but must not be citable."""
    from llm_stage import build_bundle
    df = K.load("Test3")
    leaked = set()
    for i in range(0, 200, 11):
        _, allowed = build_bundle(df.iloc[i])
        leaked |= {a for a in allowed if a.startswith(C.UNNAMED_SYSCALL_PREFIX)}
    check("no unnamed SYS_nnn artifact is citable", not leaked, str(list(leaked)[:3]))


# ------------------------------------------------------------------- retrieval
def test_retrieval_has_no_future_leakage() -> None:
    """Neighbours must come only from the training period."""
    from retrieval import NeighbourIndex
    idx = NeighbourIndex(k=6)
    latest = idx.train["ts"].max()
    earliest_test = min(K.load(s)["ts"].min() for s in K.TEST_SPLITS)
    check("retrieved neighbours all precede the test windows",
          latest < earliest_test,
          f"train max {latest.date()} vs test min {earliest_test.date()}")


def test_jaccard_matches_reference() -> None:
    """The vectorised Jaccard must equal the naive set computation exactly.

    It replaced sklearn's implementation for a ~1000x speed-up, so it needs proving
    equivalent rather than merely fast.
    """
    from retrieval import NeighbourIndex
    idx = NeighbourIndex(k=3)
    df = K.load("Test3")
    row = df.iloc[3]
    v = {c for c in idx.cols if float(row.get(c, 0)) > 0}
    worst = 0.0
    for nb in idx.query(row):
        s = set(nb["artifacts"])
        naive = len(v & s) / max(1, len(v | s))
        worst = max(worst, abs(naive - nb["similarity"]))
    check("vectorised Jaccard equals naive set Jaccard", worst < 1e-6,
          f"max deviation {worst:.2e}")


# ------------------------------------------------------------------- escalation
def test_escalation_prefix_property() -> None:
    """The set escalated at a small budget must be a prefix of a larger one.

    The whole Pareto curve is derived from a single sweep on this assumption.
    """
    from llm_stage import select_escalated
    df = K.load("Test3")
    df.attrs["split"] = "Test3"
    small = select_escalated(df, 0.05)
    large = select_escalated(df, 0.20)
    check("small budget is a prefix of the large budget",
          list(small) == list(large[:len(small)]))


def test_override_band_fitted_without_test_data() -> None:
    """The override policy must be derivable from calibration alone."""
    import inspect
    from cascade import fit_override_bands
    src = inspect.getsource(fit_override_bands)
    uses_test = any(t in src for t in ('"Test', "'Test"))
    check("override band never reads a test split", not uses_test)
    check("override band is non-empty", len(fit_override_bands()) > 0)


# ---------------------------------------------------------------- faithfulness
def test_egr_detects_injected_hallucination() -> None:
    """A synthetic ungrounded citation must be caught."""
    from faithfulness import egr
    recs = [{"cited": ["REAL_A", "FAKE_X"], "grounded": ["REAL_A"],
             "hallucinated": ["FAKE_X"], "egr": 0.5},
            {"cited": ["REAL_B"], "grounded": ["REAL_B"],
             "hallucinated": [], "egr": 1.0}]
    m = egr(recs)
    check("EGR flags an injected hallucination",
          abs(m["mean_egr"] - 0.75) < 1e-9 and m["total_hallucinated"] == 1,
          f"mean_egr={m['mean_egr']}")


def test_lenient_egr_separates_format_from_invention() -> None:
    """Packed-but-real citations must score high leniently; invented ones must not."""
    from faithfulness import lenient_egr
    allowed = {"A", "B", "C"}
    packed = [{"cited": ["A, B, C"], "grounded": [], "hallucinated": ["A, B, C"]}]
    made_up = [{"cited": ["Q, R, S"], "grounded": [], "hallucinated": ["Q, R, S"]}]
    lp = lenient_egr(packed, lambda r: allowed)
    lm = lenient_egr(made_up, lambda r: allowed)
    check("lenient EGR rescues packed real artifacts",
          lp["lenient_grounding_rate"] == 1.0,
          f"got {lp['lenient_grounding_rate']}")
    check("lenient EGR does not rescue invented artifacts",
          lm["lenient_grounding_rate"] == 0.0,
          f"got {lm['lenient_grounding_rate']}")


# -------------------------------------------------------------------- metrics
def test_metrics_against_known_confusion() -> None:
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([1, 1, 0, 1, 0, 0, 0, 0])   # tp=2 fn=1 fp=1 tn=4
    m = K.metrics(y, p)
    ok = (m["tp"] == 2 and m["fn"] == 1 and m["fp"] == 1 and m["tn"] == 4
          and abs(m["precision"] - 200 / 3) < 1e-6
          and abs(m["recall"] - 200 / 3) < 1e-6
          and abs(m["fpr"] - 20.0) < 1e-6)
    check("metrics match a hand-computed confusion matrix", ok, str(
        {k: round(m[k], 2) for k in ("precision", "recall", "f1", "fpr", "fnr")}))


def test_determinism() -> None:
    """Rebuilding the splits from the raw CSVs must reproduce what is on disk.

    The previous version read the same parquet twice and compared it with itself.
    """
    import preprocess as P
    raw = P.load_raw()
    groups = P.resolve_feature_groups(raw)
    feat_cols = groups["dynamic"] + groups["static"] + groups["aggregate"]
    raw = P.clean(raw)
    X = P.to_numeric(raw, feat_cols)
    raw = pd.concat([raw[[c for c in raw.columns if c not in feat_cols]], X], axis=1)
    rebuilt = P.build_splits(raw)
    for s in K.SPLITS:
        want = K.load(s)["sha256"].tolist()
        got = rebuilt[s]["sha256"].tolist()
        check(f"{s} split rebuilds identically from raw", got == want,
              "" if got == want else f"{len(want)} saved vs {len(got)} rebuilt")


def test_override_band_has_one_source_of_truth() -> None:
    """cascade.py and statistics.py must apply the same override policy.

    statistics.py hard-coded [0.3, 0.7) while cascade.py fits it; they agreed only on
    the reference stack.
    """
    import inspect

    import statistics as S
    from cascade import fit_override_bands
    src = inspect.getsource(S.main)
    hardcoded = re.search(r"0\.3\s*<=\s*proba", src) is not None
    check("statistics.py does not hard-code the override band", not hardcoded)
    check("statistics.py uses the fitted bands",
          "fit_override_bands" in src and "in_bands" in src)
    check("fitted override band is well formed",
          all(0.0 <= lo < hi <= 1.01 for lo, hi in fit_override_bands()))


def test_tie_breaking_is_stable() -> None:
    """Ranking by a saturating score must break ties deterministically.

    The conformal escalation score ties for 409-1,210 samples per window.
    """
    import inspect

    import conformal as CF
    import faithfulness as F
    import llm_stage as L

    def argsort_calls(src: str) -> list[str]:
        """Extract each np.argsort(...) call, matching parentheses so that nested calls
        such as np.argsort(-np.abs(x)) are not truncated."""
        calls = []
        for m in re.finditer(r"np\.argsort\(", src):
            depth, j = 0, m.end() - 1
            while j < len(src):
                if src[j] == "(":
                    depth += 1
                elif src[j] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            calls.append(src[m.start():j + 1])
        return calls

    for mod, fn in (("conformal", CF.main), ("llm_stage", L.select_escalated),
                    ("faithfulness", F.attribution_agreement)):
        src = inspect.getsource(fn)
        unstable = [s for s in argsort_calls(src) if "stable" not in s]
        check(f"{mod} sorts stably", not unstable, str(unstable[:2]))


def test_ceiling_has_one_definition() -> None:
    """Both entry points must produce the same llm_ceiling_<split>.json.

    run_experiments.py --ceiling used 350 ranked samples, run_ceiling.py 400 random ones,
    writing the same files - so whichever ran last decided the reported ceiling.
    """
    from run_ceiling import N_PER_WINDOW, ceiling_jobs
    from run_experiments import plan
    jobs = {j[0]: j for j in plan(include_ceiling=True)}
    for split in K.TEST_SPLITS:
        j = jobs.get(f"ceiling_{split}")
        ok = j is not None and j[5] == N_PER_WINDOW and j[7] == "random"
        check(f"ceiling_{split} is 100% escalation, n={N_PER_WINDOW}, random sampling",
              ok, "" if ok else str(j))
    check("run_ceiling and run_experiments agree on the ceiling jobs",
          [j for j in plan(include_ceiling=True) if j[0].startswith("ceiling_")]
          == ceiling_jobs())


def test_retrieval_ties_break_deterministically() -> None:
    """Neighbour selection must not depend on unspecified sort behaviour.

    Neighbours go into the prompt, so an unstable tie-break changes the cache key
    between numpy builds.
    """
    import inspect

    from retrieval import NeighbourIndex
    # Strip comments/docstrings: the method explains why it no longer uses argpartition.
    src = inspect.getsource(NeighbourIndex.query)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    code = re.sub(r'""".*?"""', "", code, flags=re.S)
    check("retrieval does not select neighbours with argpartition",
          "argpartition" not in code)
    check("retrieval sorts neighbours stably", 'kind="stable"' in code)

    idx = NeighbourIndex(k=6)
    df = K.load("Test3")
    row = df.iloc[3]
    a = [(n["similarity"], tuple(n["artifacts"])) for n in idx.query(row)]
    b = [(n["similarity"], tuple(n["artifacts"])) for n in idx.query(row)]
    check("repeated retrieval returns identical neighbours", a == b)
    sims = [s for s, _ in a]
    check("neighbours are returned in descending similarity",
          sims == sorted(sims, reverse=True), str([round(s, 3) for s in sims]))


def test_environment_matches_pins() -> None:
    """The stack must match requirements.txt, or no number is comparable.

    Warned rather than failed: a mismatched stack still runs and every other invariant
    still holds, but the results are not the published ones.
    """
    import envcheck
    problems = envcheck.check(verbose=False)
    if problems:
        print(f"  WARN  environment differs from requirements.txt "
              f"({len(problems)} mismatch(es)) - numbers will not match the report")
        for p in problems[:8]:
            print(f"          {p}")
    else:
        check("environment matches requirements.txt", True)


def main() -> int:
    print("=" * 78)
    print("FAITHDROID - pipeline invariant tests")
    print("=" * 78)
    groups = [
        ("splits and leakage", [test_no_temporal_leakage, test_calibration_precedes_test,
                                test_prevalence_pinned, test_no_label_leakage_in_features]),
        ("evidence bundle", [test_bundle_is_closed_and_true,
                             test_bundle_excludes_unnamed_syscalls]),
        ("retrieval", [test_retrieval_has_no_future_leakage, test_jaccard_matches_reference,
                       test_retrieval_ties_break_deterministically]),
        ("escalation and override", [test_escalation_prefix_property,
                                     test_override_band_fitted_without_test_data,
                                     test_override_band_has_one_source_of_truth,
                                     test_ceiling_has_one_definition]),
        ("faithfulness", [test_egr_detects_injected_hallucination,
                          test_lenient_egr_separates_format_from_invention]),
        ("metrics", [test_metrics_against_known_confusion]),
        ("reproducibility", [test_determinism, test_tie_breaking_is_stable,
                             test_environment_matches_pins]),
    ]
    for title, fns in groups:
        print(f"\n[{title}]")
        for fn in fns:
            try:
                fn()
            except Exception as e:
                check(fn.__name__, False, f"{type(e).__name__}: {e}")

    print(f"\n{'=' * 78}\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
