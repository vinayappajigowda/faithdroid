"""Stage 3 - Mondrian inductive conformal predictor used as an escalation gate.

Transcend and Transcendent (USENIX) use conformal evaluation to *reject* drifting
malware samples. Here the same statistical machinery is repurposed: instead of
discarding a low-credibility sample, we **route** it to an expensive LLM reasoner.
That re-purposing is the contribution of this stage.

Definitions (class-conditional / Mondrian, so the guarantee holds per class even at
10% prevalence):

    nonconformity   alpha(x, y) = 1 - p_hat(y | x)
    p-value         p_y = |{i in calib_y : alpha_i >= alpha(x, y)}| + 1
                          -------------------------------------------
                                        n_y + 1

    credibility = max(p_0, p_1)      how well the best label is supported
    confidence  = 1 - min(p_0, p_1)  how firmly the other label is excluded

Two distinct failure modes justify escalation, and the p-values separate them:

    ambiguous  min(p_0, p_1) is large  -> both labels are supported by calibration
    novel      max(p_0, p_1) is small  -> neither label is; the sample is unlike
                                          anything seen, i.e. drift

so the routing score is  escalation_score = min(p) + (1 - max(p)),
which is large under either condition. For budget-controlled routing we escalate the
top-k% by this score; separately we report conformal set sizes at nominal significance
to verify the gate is statistically meaningful and not just a heuristic ranking.

Run:  python3 code/conformal.py
"""
from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier

import common as K
import config as C


class MondrianConformal:
    """Class-conditional inductive conformal predictor over a probabilistic model."""

    def __init__(self, model):
        self.model = model
        self.alpha_cal: dict[int, np.ndarray] = {}

    def calibrate(self, X, y) -> "MondrianConformal":
        proba = self.model.predict_proba(X)
        for cls in (0, 1):
            mask = y == cls
            # Nonconformity of a calibration point w.r.t. its own true label.
            self.alpha_cal[cls] = np.sort(1.0 - proba[mask, cls])
        return self

    def p_values(self, X) -> np.ndarray:
        """Return an (n, 2) array of p-values, column j = p-value for label j."""
        proba = self.model.predict_proba(X)
        out = np.empty((len(X), 2), dtype=float)
        for cls in (0, 1):
            a_cal = self.alpha_cal[cls]
            a_test = 1.0 - proba[:, cls]
            # count of calibration alphas >= alpha_test, via the sorted array
            ge = len(a_cal) - np.searchsorted(a_cal, a_test, side="left")
            out[:, cls] = (ge + 1) / (len(a_cal) + 1)
        return out

    # ------------------------------------------------------------------ derived
    @staticmethod
    def credibility(p: np.ndarray) -> np.ndarray:
        return p.max(axis=1)

    @staticmethod
    def confidence(p: np.ndarray) -> np.ndarray:
        return 1.0 - p.min(axis=1)

    @staticmethod
    def escalation_score(p: np.ndarray) -> np.ndarray:
        """Large when the sample is ambiguous OR novel - both warrant escalation."""
        return p.min(axis=1) + (1.0 - p.max(axis=1))

    @staticmethod
    def prediction_set_size(p: np.ndarray, eps: float) -> np.ndarray:
        return (p > eps).sum(axis=1)


def fit_base(Xtr, ytr) -> XGBClassifier:
    """XGBoost was the strongest baseline, so it is the Stage-2 detector."""
    m = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=(1 - C.TARGET_PREVALENCE) / C.TARGET_PREVALENCE,
        eval_metric="logloss", n_jobs=-1, random_state=C.SEED, tree_method="hist")
    m.fit(Xtr, ytr)
    return m


def main() -> None:
    print("=" * 78)
    print("FAITHDROID - Stage 3: Mondrian conformal escalation gate")
    print("=" * 78)

    Xtr, ytr = K.xy(K.load("train"))
    Xca, yca = K.xy(K.load("calib"))
    base = fit_base(Xtr, ytr)
    cp = MondrianConformal(base).calibrate(Xca, yca)
    print(f"\ncalibrated on n={len(yca):,}  "
          f"(class 0: {len(cp.alpha_cal[0]):,}, class 1: {len(cp.alpha_cal[1]):,})")

    # ------------------------------------------------ validity check on calibration
    # An inductive conformal predictor must be valid on exchangeable data. The
    # calibration set is exchangeable with the training period, so empirical error
    # there should track the nominal significance. Test windows are NOT exchangeable
    # (that is the drift we are studying), so we expect degradation - and quantifying
    # that degradation is precisely what the gate exploits.
    print("\n[validity] empirical error rate of the conformal prediction set")
    print(f"  {'split':<8s} {'eps':>5s} {'err':>7s} {'|set|=0':>8s} "
          f"{'|set|=1':>8s} {'|set|=2':>8s}")
    # Persisted as well as printed: the report quotes these figures, and a number that
    # exists only on stdout cannot be regenerated into the prose or checked afterwards.
    validity: dict[str, dict] = {}
    for split in ["calib"] + K.TEST_SPLITS:
        X, y = K.xy(K.load(split))
        p = cp.p_values(X)
        validity[split] = {}
        for eps in (0.05, C.CONFORMAL_EPSILON):
            in_set = p > eps
            err = 100 * (1 - in_set[np.arange(len(y)), y].mean())
            sz = in_set.sum(axis=1)
            validity[split][f"{eps:.2f}"] = {
                "error_pct": float(err),
                "empty_pct": float(100 * (sz == 0).mean()),
                "single_pct": float(100 * (sz == 1).mean()),
                "both_pct": float(100 * (sz == 2).mean()),
            }
            print(f"  {split:<8s} {eps:5.2f} {err:6.2f}% "
                  f"{100 * (sz == 0).mean():7.1f}% {100 * (sz == 1).mean():7.1f}% "
                  f"{100 * (sz == 2).mean():7.1f}%")

    # ---------------------------------------------- does the gate select real errors?
    # The gate is only useful if the samples it flags are disproportionately the ones
    # the cheap detector gets wrong. This is the key diagnostic for the whole thesis.
    print("\n[selectivity] base-detector error rate inside vs outside the escalated set")
    print(f"  {'split':<8s} {'budget':>7s} {'err|escalated':>14s} "
          f"{'err|kept':>10s} {'lift':>7s} {'%errors caught':>15s}")
    gate_stats: dict[str, dict] = {}
    for split in K.TEST_SPLITS:
        X, y = K.xy(K.load(split))
        p = cp.p_values(X)
        score = MondrianConformal.escalation_score(p)
        pred = base.predict(X)
        wrong = pred != y
        gate_stats[split] = {}
        for budget in (0.05, 0.10, 0.20, 0.30):
            k = max(1, int(round(budget * len(y))))
            esc = np.zeros(len(y), dtype=bool)
            # Stable, matching llm_stage. The score saturates under drift (F3): 409-1,210
            # samples per window tie, so an unstable sort shifts the top-k boundary.
            esc[np.argsort(-score, kind="stable")[:k]] = True
            e_in = wrong[esc].mean() if esc.any() else 0.0
            e_out = wrong[~esc].mean() if (~esc).any() else 0.0
            caught = wrong[esc].sum() / max(1, wrong.sum())
            lift = e_in / e_out if e_out > 0 else float("inf")
            gate_stats[split][f"{budget:.2f}"] = {
                "err_escalated": 100 * e_in, "err_kept": 100 * e_out,
                "lift": lift, "errors_caught": 100 * caught,
                "n_escalated": int(esc.sum()),
            }
            print(f"  {split:<8s} {100 * budget:6.0f}% {100 * e_in:13.1f}% "
                  f"{100 * e_out:9.1f}% {lift:7.2f} {100 * caught:14.1f}%")

    K.save_json({"selectivity": gate_stats, "validity": validity},
                C.RESULTS / "conformal_gate.json")

    # persist p-values so later stages need not refit
    store = {}
    for split in K.TEST_SPLITS:
        X, y = K.xy(K.load(split))
        p = cp.p_values(X)
        store[split] = {
            "p0": p[:, 0].tolist(), "p1": p[:, 1].tolist(),
            "proba": base.predict_proba(X)[:, 1].tolist(),
            "pred": base.predict(X).tolist(), "y": y.tolist(),
            "escalation_score": MondrianConformal.escalation_score(p).tolist(),
            "credibility": MondrianConformal.credibility(p).tolist(),
            "confidence": MondrianConformal.confidence(p).tolist(),
        }
    K.save_json(store, C.RESULTS / "conformal_pvalues.json")


if __name__ == "__main__":
    main()
