"""Stage 7 - uncertainty quantification for every headline number.

The evaluation so far reports point estimates. Several of them rest on small samples --
163 escalated applications on Test1, 105 completed responses from the 14B model -- and a
difference of a few F1 points between configurations may not be distinguishable from
sampling noise. Reporting such differences without intervals would overstate what the
experiments establish.

Two procedures are used:

  Bootstrap CI      Resample applications with replacement, recompute the metric, and
                    take percentile bounds. Applied to F1 and to the cascade's gain over
                    the base detector, which is the quantity the thesis actually claims.

  McNemar's test    For paired predictions on the same samples -- base detector versus
                    cascade -- the appropriate test is McNemar's, since the two systems
                    are evaluated on identical inputs and their errors are correlated.
                    An unpaired test would be invalid here.

Run:  python3 code/statistics.py
"""
from __future__ import annotations

import json

import numpy as np
from scipy.stats import binomtest

import common as K
import config as C

N_BOOT = 10_000

# Each bootstrap reseeds from a fixed seed rather than sharing a module-level generator.
# With a shared generator the intervals depend on how many other bootstraps ran first, so
# the same experiment would report slightly different bounds depending on call order -
# which would in turn make the report-verification harness flap. Reseeding makes every
# interval a deterministic function of its inputs alone.
def _rng() -> np.random.Generator:
    return np.random.default_rng(C.SEED)


def _f1(y, p) -> float:
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    if tp == 0:
        return 0.0
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    return 200 * prec * rec / (prec + rec)


def bootstrap_ci(y, pred, stat=_f1, n_boot: int = N_BOOT, alpha: float = 0.05):
    """Percentile bootstrap interval for a metric over applications."""
    y, pred = np.asarray(y), np.asarray(pred)
    n = len(y)
    rng = _rng()
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = stat(y[idx], pred[idx])
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat(y, pred)), float(lo), float(hi)


def bootstrap_delta_ci(y, pred_a, pred_b, n_boot: int = N_BOOT, alpha: float = 0.05):
    """Interval for the *paired* difference stat(b) - stat(a).

    Both systems are resampled on the same bootstrap indices, which preserves the
    pairing and is what makes the interval a statement about the difference rather than
    about two independent estimates.
    """
    y = np.asarray(y)
    pred_a, pred_b = np.asarray(pred_a), np.asarray(pred_b)
    n = len(y)
    rng = _rng()
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[i] = _f1(y[idx], pred_b[idx]) - _f1(y[idx], pred_a[idx])
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(_f1(y, pred_b) - _f1(y, pred_a)), float(lo), float(hi)


def mcnemar(y, pred_a, pred_b) -> dict:
    """Exact McNemar test on the discordant pairs."""
    y = np.asarray(y)
    a_ok = np.asarray(pred_a) == y
    b_ok = np.asarray(pred_b) == y
    b01 = int((a_ok & ~b_ok).sum())   # a right, b wrong
    b10 = int((~a_ok & b_ok).sum())   # a wrong, b right
    n = b01 + b10
    p = binomtest(b10, n, 0.5).pvalue if n else 1.0
    return {"a_only_correct": b01, "b_only_correct": b10,
            "discordant": n, "p_value": float(p)}


def proportion_ci(k: int, n: int, alpha: float = 0.05):
    """Wilson interval for a proportion - used for EGR-style rates."""
    if n == 0:
        return 0.0, 0.0, 0.0
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return float(p), float(max(0, c - h)), float(min(1, c + h))


def main() -> None:
    print("=" * 88)
    print("FAITHDROID - Stage 7: confidence intervals and significance tests")
    print("=" * 88)

    casc = {c["tag"]: c for c in json.loads((C.RESULTS / "cascade.json").read_text())}
    pv = json.loads((C.RESULTS / "conformal_pvalues.json").read_text())
    out: dict = {}

    # The override band is fitted in cascade.py; never restate it here. This module used
    # to hard-code [0.3, 0.7), which matched the fit only by coincidence on one stack.
    from cascade import fit_override_bands, in_bands
    bands = fit_override_bands()
    print(f"  override bands (fitted on calibration): {bands}")
    out["override_bands"] = [list(b) for b in bands]

    print("\n[1] Cascade gain over the base detector (paired bootstrap + McNemar)")
    print(f"  {'window':<10s} {'base F1':>8s} {'banded F1':>10s} {'gain':>7s} "
          f"{'95% CI':>18s} {'McNemar p':>10s} {'sig?':>5s}")
    for tag, split in (("main_Test1", "Test1"), ("main_Test2", "Test2"),
                       ("main_Test3_b20", "Test3")):
        if tag not in casc:
            continue
        d = json.loads((C.RESULTS / f"llm_{tag}.json").read_text())
        y = np.asarray(pv[split]["y"])
        base = np.asarray(pv[split]["pred"])
        proba = np.asarray(pv[split]["proba"])

        # rebuild the banded-override prediction using the *fitted* bands
        banded = base.copy()
        rec_idx = np.array([r["row_index"] for r in d["records"]], dtype=int)
        rec_pred = np.array([r["llm_verdict"] for r in d["records"]], dtype=int)
        keep = in_bands(proba[rec_idx], bands)
        banded[rec_idx[keep]] = rec_pred[keep]

        gain, lo, hi = bootstrap_delta_ci(y, base, banded)
        mc = mcnemar(y, base, banded)
        sig = "yes" if mc["p_value"] < 0.05 else "no"
        print(f"  {split:<10s} {_f1(y, base):8.2f} {_f1(y, banded):10.2f} "
              f"{gain:+7.2f} [{lo:+6.2f},{hi:+6.2f}] {mc['p_value']:10.4f} {sig:>5s}")
        out[split] = {"base_f1": _f1(y, base), "banded_f1": _f1(y, banded),
                      "gain": gain, "ci95": [lo, hi], "mcnemar": mc}

    print("\n[2] Baseline F1 with bootstrap intervals")
    from baselines import build_models
    Xtr, ytr = K.xy(K.load("train"))
    tests = {s: K.xy(K.load(s)) for s in K.TEST_SPLITS}
    print(f"  {'model':<16s} " + "".join(f"{s:>24s}" for s in K.TEST_SPLITS))
    out["baselines"] = {}
    for name, model in build_models().items():
        model.fit(Xtr, ytr)
        row = f"  {name:<16s} "
        out["baselines"][name] = {}
        for s in K.TEST_SPLITS:
            Xte, yte = tests[s]
            f1, lo, hi = bootstrap_ci(yte, model.predict(Xte), n_boot=2000)
            row += f"{f1:7.2f} [{lo:5.1f},{hi:5.1f}]  "
            out["baselines"][name][s] = {"f1": f1, "ci95": [lo, hi]}
        print(row)

    print("\n[3] Grounding rates with Wilson intervals")
    faith = json.loads((C.RESULTS / "faithfulness.json").read_text())
    print(f"  {'config':<30s} {'n cit.':>7s} {'EGR':>7s} {'95% CI':>16s}")
    for f in sorted(faith, key=lambda x: -x["n"]):
        if f["n"] < 100:
            continue
        E = f["EGR"]
        n_cit = E["total_citations"]
        k = n_cit - E["total_hallucinated"]
        p, lo, hi = proportion_ci(k, n_cit)
        print(f"  {f['tag']:<30s} {n_cit:7d} {p:7.3f} [{lo:.3f},{hi:.3f}]")
        out.setdefault("grounding", {})[f["tag"]] = {
            "rate": p, "ci95": [lo, hi], "n_citations": n_cit}

    K.save_json(out, C.RESULTS / "statistics.json")


if __name__ == "__main__":
    main()
