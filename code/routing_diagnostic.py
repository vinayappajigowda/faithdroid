"""Diagnostic - which uncertainty signal actually selects the base detector's errors?

Stage 3's first attempt used conformal *novelty* (1 - max p) as the routing score and it
was anti-selective (lift < 1): escalated samples were classified more accurately than
retained ones. Rather than assume a fix, this script measures every plausible routing
signal against the only criterion that matters - does the escalated set concentrate the
errors the cheap detector makes?

Reported per signal and budget:
    lift            P(error | escalated) / P(error | kept).  >1 means useful.
    errors_caught   share of all errors that land inside the escalated set.
    oracle_lift     the same quantity for a perfect router, as an upper bound.

Run:  python3 code/routing_diagnostic.py
"""
from __future__ import annotations

import numpy as np

import common as K
import config as C
from conformal import MondrianConformal, fit_base


def signals(p: np.ndarray, proba: np.ndarray) -> dict[str, np.ndarray]:
    """Candidate routing scores; higher = escalate sooner."""
    eps = 1e-12
    return {
        # --- conformal-derived
        "conformal_novelty": 1.0 - p.max(axis=1),
        "conformal_ambiguity": p.min(axis=1),
        "conformal_combined": p.min(axis=1) + (1.0 - p.max(axis=1)),
        "conformal_ratio": p.min(axis=1) / (p.max(axis=1) + eps),
        # --- probability-derived (classic uncertainty sampling)
        "prob_margin": 1.0 - np.abs(proba - 0.5) * 2.0,
        "prob_entropy": -(proba * np.log(proba + eps)
                          + (1 - proba) * np.log(1 - proba + eps)),
        # --- directional: suspicion of a missed positive. At 10% prevalence most
        # errors are false negatives, so ranking by raw malware probability may
        # concentrate them even though it ignores uncertainty entirely.
        "prob_malware": proba,
        # --- hybrid: uncertainty, tie-broken toward the positive class
        "margin_x_malware": (1.0 - np.abs(proba - 0.5) * 2.0) * (0.5 + proba),
    }


def main() -> None:
    print("=" * 96)
    print("ROUTING DIAGNOSTIC - does the gate concentrate the base detector's errors?")
    print("=" * 96)

    Xtr, ytr = K.xy(K.load("train"))
    Xca, yca = K.xy(K.load("calib"))
    base = fit_base(Xtr, ytr)
    cp = MondrianConformal(base).calibrate(Xca, yca)

    budgets = [0.05, 0.10, 0.20, 0.30]
    agg: dict[str, dict[float, list[float]]] = {}

    for split in K.TEST_SPLITS:
        X, y = K.xy(K.load(split))
        p = cp.p_values(X)
        proba = base.predict_proba(X)[:, 1]
        pred = base.predict(X)
        wrong = pred != y
        n_err = int(wrong.sum())

        print(f"\n{'=' * 96}\n{split}: n={len(y):,}  base errors={n_err} "
              f"({100 * wrong.mean():.1f}%)  "
              f"FN={int(((pred == 0) & (y == 1)).sum())} "
              f"FP={int(((pred == 1) & (y == 0)).sum())}\n{'=' * 96}")
        header = f"  {'signal':<22s}" + "".join(f"{f'b={int(100*b)}%':>22s}" for b in budgets)
        print(header)
        print(f"  {'':<22s}" + "".join(f"{'lift / caught':>22s}" for _ in budgets))

        for name, score in signals(p, proba).items():
            row = f"  {name:<22s}"
            for b in budgets:
                k = max(1, int(round(b * len(y))))
                esc = np.zeros(len(y), dtype=bool)
                esc[np.argsort(-score, kind="stable")[:k]] = True
                e_in = wrong[esc].mean()
                e_out = wrong[~esc].mean() if (~esc).any() else 0.0
                lift = e_in / e_out if e_out > 0 else float("inf")
                caught = 100 * wrong[esc].sum() / max(1, n_err)
                row += f"{lift:12.2f} /{caught:6.1f}%"
                agg.setdefault(name, {}).setdefault(b, []).append(lift)
            print(row)

        # upper bound: a perfect router escalates errors first
        row = f"  {'ORACLE (upper bound)':<22s}"
        for b in budgets:
            k = max(1, int(round(b * len(y))))
            caught = 100 * min(k, n_err) / max(1, n_err)
            e_in = min(k, n_err) / k
            rem = n_err - min(k, n_err)
            e_out = rem / (len(y) - k) if len(y) - k else 0.0
            lift = e_in / e_out if e_out > 0 else float("inf")
            row += f"{lift:12.2f} /{caught:6.1f}%"
        print(row)

    print(f"\n{'=' * 96}\nMEAN LIFT ACROSS THE THREE DRIFT WINDOWS "
          f"(>1.0 = selects errors better than chance)\n{'=' * 96}")
    print(f"  {'signal':<22s}" + "".join(f"{f'b={int(100*b)}%':>12s}" for b in budgets))
    ranked = sorted(agg.items(), key=lambda kv: -np.mean(kv[1][0.10]))
    for name, per_b in ranked:
        print(f"  {name:<22s}" + "".join(
            f"{np.mean(per_b[b]):12.2f}" for b in budgets))

    K.save_json({n: {str(b): v for b, v in d.items()} for n, d in agg.items()},
                C.RESULTS / "routing_diagnostic.json")


if __name__ == "__main__":
    main()
