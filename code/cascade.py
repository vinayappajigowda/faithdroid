"""Stage 6 - assemble the cascade and measure the cost-accuracy trade-off.

Final decision rule:

    prediction(x) = LLM verdict          if x was escalated
                    base detector output otherwise

so the cheap detector decides the confident majority and the LLM is spent only on the
routed minority. Reported alongside every configuration:

  escalation_rate    share of inputs that reached the LLM
  usd_per_1k_apps    token-priced cost of the LLM stage, so the operating point can be
                     compared with LAMD's $0.199/APK and TraceRAG's $6.00/app. Local
                     inference costs $0 in fees; this figure prices the *same tokens* at
                     GPT-4o-mini rates to make the comparison like-for-like, and both
                     numbers are reported.
  wall_clock_s       measured local latency, which is the real cost here

Run:  python3 code/cascade.py
"""
from __future__ import annotations

import json

import numpy as np

import common as K
import config as C

# GPT-4o-mini list prices used by LAMD, for a like-for-like cost comparison.
USD_PER_M_INPUT = 0.15
USD_PER_M_OUTPUT = 0.60

# Fitted once in main() from the calibration split; see fit_override_bands().
OVERRIDE_BANDS: list[tuple[float, float]] | None = None
# Per-band calibration support and accuracy, persisted so the prose can quote it.
BAND_FIT: list[dict] = []


def load_base(split: str) -> dict:
    pv = json.loads((C.RESULTS / "conformal_pvalues.json").read_text())[split]
    return {"pred": np.asarray(pv["pred"]), "y": np.asarray(pv["y"]),
            "proba": np.asarray(pv["proba"])}


# --------------------------------------------------------------- override policy
# Measured on Test3: a blind override degrades F1 badly as the budget grows, because
# beyond a small budget the escalated set starts including samples the cheap detector
# handles well, and the LLM is worse than what it replaces. The fix is to override only
# where the detector is *known* to be unreliable.
#
# The unreliable region is fitted on the CALIBRATION split alone - never on test - so
# the rule carries no leakage. Calibration accuracy by predicted-probability band is
# 0.994 in [0.0,0.3), 0.629 in [0.3,0.5), 0.469 in [0.5,0.7) and 0.976 in [0.7,1.0],
# so the middle bands are where a second opinion is warranted.
BAND_EDGES = [0.0, 0.3, 0.5, 0.7, 1.01]
BAND_RELIABILITY_THRESHOLD = 0.70

# Band accuracy is only meaningful with enough calibration support. [0.3,0.5) holds ~27 of
# 10,107 samples and sits within 0.005 of the threshold, so which side it falls on flips
# with the library stack. Thin bands are flagged rather than presented as a firm boundary.
MIN_BAND_SUPPORT = 100


def fit_override_bands(verbose: bool = False) -> list[tuple[float, float]]:
    """Return the probability bands in which the base detector is unreliable.

    Fitted on the calibration split only.
    """
    import common as _K
    from conformal import fit_base as _fit
    Xtr, ytr = _K.xy(_K.load("train"))
    model = _fit(Xtr, ytr)
    Xca, yca = _K.xy(_K.load("calib"))
    p = model.predict_proba(Xca)[:, 1]
    pred = (p >= 0.5).astype(int)

    bands = []
    BAND_FIT.clear()
    for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        BAND_FIT.append({"lo": lo, "hi": hi, "n": int(m.sum()),
                         "accuracy": float((pred[m] == yca[m]).mean())})
        acc = float((pred[m] == yca[m]).mean())
        n = int(m.sum())
        thin = n < MIN_BAND_SUPPORT
        margin = abs(acc - BAND_RELIABILITY_THRESHOLD)
        if verbose:
            note = ""
            if acc < BAND_RELIABILITY_THRESHOLD:
                note = "   <- unreliable, override here"
            if thin:
                note += (f"   [LOW SUPPORT n={n}, margin {margin:.3f} "
                         f"- boundary not robust]")
            print(f"    calib band [{lo:.1f},{hi:.1f}): n={n:5d} acc={acc:.3f}{note}")
        if acc < BAND_RELIABILITY_THRESHOLD:
            bands.append((lo, hi))
    return bands


def in_bands(proba: np.ndarray, bands: list[tuple[float, float]]) -> np.ndarray:
    m = np.zeros(len(proba), dtype=bool)
    for lo, hi in bands:
        m |= (proba >= lo) & (proba < hi)
    return m


def token_cost(records: list[dict]) -> dict:
    pin = sum(r.get("prompt_eval_count", 0) for r in records)
    pout = sum(r.get("eval_count", 0) for r in records)
    usd = pin / 1e6 * USD_PER_M_INPUT + pout / 1e6 * USD_PER_M_OUTPUT
    return {"prompt_tokens": pin, "output_tokens": pout,
            "equiv_usd_gpt4o_mini": usd,
            "usd_per_app_equiv": usd / max(1, len(records))}


def combine(path, prefix: int | None = None, tag_suffix: str = "") -> dict | None:
    """Assemble the cascade for one LLM run.

    `prefix` truncates the escalated set to its first N records. Because escalation is
    a deterministic ranking by uncertainty, the first N of a larger run is exactly the
    set a run at the smaller budget would have selected - so the whole Pareto curve
    comes from one sweep.
    """
    d = json.loads(path.read_text())
    recs = d.get("records", [])
    if prefix is not None:
        recs = recs[:prefix]
    if not recs:
        return None
    split = d["split"]
    base = load_base(split)
    y = base["y"]
    pred = base["pred"].copy()

    esc_idx = np.array([r["row_index"] for r in recs], dtype=int)
    llm_pred = np.array([r["llm_verdict"] for r in recs], dtype=int)

    # How often does each stage get the escalated samples right?
    base_on_esc = K.metrics(y[esc_idx], base["pred"][esc_idx])
    llm_on_esc = K.metrics(y[esc_idx], llm_pred)

    pred[esc_idx] = llm_pred
    casc = K.metrics(y, pred, n_llm_calls=len(esc_idx), n_total=len(y))

    # ---- banded override: replace the base verdict ONLY inside the unreliable region
    bands = OVERRIDE_BANDS if OVERRIDE_BANDS is not None else []
    pred_b = base["pred"].copy()
    keep = in_bands(base["proba"][esc_idx], bands)
    b_idx, b_pred = esc_idx[keep], llm_pred[keep]
    pred_b[b_idx] = b_pred
    banded = K.metrics(y, pred_b, n_llm_calls=int(keep.sum()), n_total=len(y))
    banded_detail = {
        "n_overridden": int(keep.sum()),
        "override_rate": 100 * keep.sum() / len(y),
        "base_on_overridden": K.metrics(y[b_idx], base["pred"][b_idx])["accuracy"]
                              if len(b_idx) else 0.0,
        "llm_on_overridden": K.metrics(y[b_idx], b_pred)["accuracy"]
                             if len(b_idx) else 0.0,
    }

    cost = token_cost(recs)
    fresh = [r for r in recs if r.get("latency_s", 0) > 0]
    return {
        "tag": path.stem.replace("llm_", "") + tag_suffix,
        "split": split, "model": d["model"], "mode": d["mode"],
        "requested_budget": d["budget"],
        "n_escalated": len(esc_idx), "n_total": len(y),
        "base_only": K.metrics(y, base["pred"]),
        "cascade": casc,
        "cascade_banded": banded,
        "banded_detail": banded_detail,
        "base_on_escalated": base_on_esc,
        "llm_on_escalated": llm_on_esc,
        "delta_f1_vs_base": casc["f1"] - K.metrics(y, base["pred"])["f1"],
        "delta_f1_banded_vs_base": banded["f1"] - K.metrics(y, base["pred"])["f1"],
        "cost": cost,
        "usd_per_1k_apps_equiv": cost["equiv_usd_gpt4o_mini"] / max(1, len(y)) * 1000,
        "mean_latency_s": float(np.mean([r["latency_s"] for r in fresh])) if fresh else 0.0,
        "wall_clock_s": d.get("wall_clock_s", 0.0),
    }


def ceiling_reference() -> dict:
    """LLM-only reference point: the operating point of TraceRAG and LAMD.

    Estimated on a uniform random subsample of each window rather than the full window,
    so the figure is an estimate; the interval is reported with it. Random rather than
    ranked selection matters here - the ranked set is deliberately biased toward
    uncertain samples, so a ranked prefix would understate what the reasoner achieves on
    the population.
    """
    from statistics import bootstrap_ci   # local import: only needed for this block
    out = {}
    for split in K.TEST_SPLITS:
        p = C.RESULTS / f"llm_ceiling_{split}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        recs = d["records"]
        base = load_base(split)
        idx = np.array([r["row_index"] for r in recs], dtype=int)
        y = base["y"][idx]
        llm = np.array([r["llm_verdict"] for r in recs], dtype=int)

        from statistics import bootstrap_delta_ci, mcnemar
        f1, lo, hi = bootstrap_ci(y, llm, n_boot=2000)
        bf1, blo, bhi = bootstrap_ci(y, base["pred"][idx], n_boot=2000)
        # Paired comparison: both systems judged the same applications, so the
        # difference must be tested pairwise rather than by comparing two intervals.
        delta, dlo, dhi = bootstrap_delta_ci(y, base["pred"][idx], llm, n_boot=2000)
        mc = mcnemar(y, base["pred"][idx], llm)
        cost = token_cost(recs)
        # Cost of running the LLM on 100% of the window, extrapolated per application.
        per_app = cost["equiv_usd_gpt4o_mini"] / max(1, len(recs))
        out[split] = {
            "n_sampled": len(recs), "n_window": len(base["y"]),
            "llm_only": K.metrics(y, llm),
            "llm_only_f1_ci": [lo, hi],
            "base_on_same_subsample": K.metrics(y, base["pred"][idx]),
            "base_f1_ci": [blo, bhi],
            "delta_llm_minus_base": delta,
            "delta_ci95": [dlo, dhi],
            "mcnemar": mc,
            "usd_per_1k_apps_equiv": per_app * 1000,
            "mean_egr": d.get("mean_egr", 0.0),
        }
    return out


def main() -> None:
    print("=" * 104)
    print("FAITHDROID - Stage 6: cascade assembly and cost-accuracy trade-off")
    print("=" * 104)

    global OVERRIDE_BANDS
    print("\n[override policy] fitting the unreliable region on CALIBRATION only:")
    OVERRIDE_BANDS = fit_override_bands(verbose=True)
    print(f"  -> override bands: {OVERRIDE_BANDS}")
    K.save_json({"bands": [list(b) for b in OVERRIDE_BANDS],
                 "threshold": BAND_RELIABILITY_THRESHOLD,
                 "min_support": MIN_BAND_SUPPORT,
                 "per_band": BAND_FIT},
                C.RESULTS / "override_bands.json")

    rows = []
    for p in sorted(C.RESULTS.glob("llm_*.json")):
        r = combine(p)
        if r:
            rows.append(r)
        # Derive the Pareto curve from the largest run on each split by taking
        # prefixes of its deterministic escalation ranking.
        d = json.loads(p.read_text())
        if d.get("mode") == "context" and d.get("budget", 0) >= 0.20:
            n_total = d["n_total"]
            for b in C.ESCALATION_BUDGETS:
                k = int(round(b * n_total))
                if 0 < k < len(d["records"]):
                    sub = combine(p, prefix=k, tag_suffix=f"@b{int(100 * b)}")
                    if sub:
                        sub["requested_budget"] = b
                        sub["derived"] = True
                        rows.append(sub)
    if not rows:
        print("no LLM results yet")
        return

    hdr = (f"\n{'config':<30s} {'esc%':>6s} {'base':>7s} "
           f"{'BLIND override':>21s} | {'BANDED override':>23s}")
    print(hdr)
    print(f"{'':<30s} {'':>6s} {'F1':>7s} {'F1':>7s} {'dF1':>7s} {'llm@esc':>6s} | "
          f"{'ovr%':>5s} {'F1':>7s} {'dF1':>7s} {'llm@ovr':>7s}")
    print("-" * 108)
    for r in sorted(rows, key=lambda x: (x["split"], x["requested_budget"])):
        bd = r["banded_detail"]
        print(f"{r['tag']:<30s} {r['cascade']['escalation_rate']:5.1f}% "
              f"{r['base_only']['f1']:7.2f} "
              f"{r['cascade']['f1']:7.2f} {r['delta_f1_vs_base']:+7.2f} "
              f"{r['llm_on_escalated']['accuracy']:5.1f}% | "
              f"{bd['override_rate']:4.1f}% "
              f"{r['cascade_banded']['f1']:7.2f} "
              f"{r['delta_f1_banded_vs_base']:+7.2f} "
              f"{bd['llm_on_overridden']:6.1f}%")

    ceil = ceiling_reference()
    if ceil:
        print(f"\n{'=' * 104}")
        print("LLM-ONLY CEILING (100% escalation - the TraceRAG / LAMD operating point)")
        print("=" * 104)
        print(f"  {'window':<7s} {'n':>4s} {'LLM-only F1':>20s} {'base F1':>20s} "
              f"{'LLM-base':>9s} {'95% CI':>16s} {'p':>7s}")
        for s, c in ceil.items():
            lo, hi = c["llm_only_f1_ci"]
            blo, bhi = c["base_f1_ci"]
            dlo, dhi = c["delta_ci95"]
            print(f"  {s:<7s} {c['n_sampled']:4d} "
                  f"{c['llm_only']['f1']:7.2f} [{lo:4.1f},{hi:4.1f}] "
                  f"{c['base_on_same_subsample']['f1']:7.2f} [{blo:4.1f},{bhi:4.1f}] "
                  f"{c['delta_llm_minus_base']:+9.2f} [{dlo:+6.2f},{dhi:+6.2f}] "
                  f"{c['mcnemar']['p_value']:7.4f}")
        K.save_json(ceil, C.RESULTS / "ceiling.json")

    print("\nReference operating points from the literature:")
    print(f"  TraceRAG  $6.000 per app  = $6000.00 per 1k apps  (100% escalation)")
    print(f"  LAMD      $0.199 per app  = $ 199.00 per 1k apps  (100% escalation)")

    K.save_json(rows, C.RESULTS / "cascade.json")


if __name__ == "__main__":
    main()
