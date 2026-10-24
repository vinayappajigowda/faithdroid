"""Stage 5 - automatic faithfulness metrics for LLM malware explanations.

This is the thesis's central contribution. Prior work validates LLM malware
explanations either not at all, or with expert human studies that do not scale
(TraceRAG: Google Android Security engineers over 24 APKs, 65.52% behaviour
alignment). MalEval formalises the evaluation but still needs human-written audit
reports as ground truth. The XAI-for-malware survey states outright that evaluating
explanation faithfulness remains open, and the Delft SoK reports that security XAI
papers run user studies in only 14% of cases - while noting that most explanation
criteria "can be evaluated without human involvement".

Three metrics, none of which needs a human label:

  EGR  Evidence Grounding Rate
       Fraction of cited artifacts that genuinely occur in the app's feature vector.
       Computable because the citation vocabulary is closed by construction. Directly
       measures hallucination.

  CFN  Counterfactual Necessity
       Zero out the cited artifacts in the feature vector and re-score with the
       Stage-2 detector. If the cited evidence is truly load-bearing, the score should
       move toward the opposite class. Follows the fidelity-by-perturbation precedent
       the SoK documents (Wickramasinghe et al.; Lin et al.), so it is an application
       of accepted methodology rather than an invented measure.
       Reported as (a) mean absolute score shift and (b) decision-flip rate.

  AA   Attribution Agreement
       Overlap between the artifacts the LLM cites and the top-k features by |SHAP|
       for the same sample under the Stage-2 detector. This is the bridge between the
       two disjoint literatures: the XAI side owns the attribution machinery, the LLM
       side owns the readable narrative, and nobody has used the former to audit the
       latter. Reported as precision@k against the SHAP top-k and as a random baseline
       for reference, since overlap must beat chance to mean anything.

Run:  python3 code/faithfulness.py
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import shap

import common as K
import config as C
from conformal import fit_base


# ------------------------------------------------------------------------- EGR
def _split_citation(s: str) -> list[str]:
    """Split a citation field that packs several artifact names into one string.

    phi4:14b exposed a flaw in the strict metric: it emits comma-separated lists inside a
    single array element ("ACCESS_NETWORK_STATE, ACCESS_WIFI_STATE, INTERNET, ..."), which
    the schema's length cap then truncates. Strict EGR scores every such element as
    ungrounded, so a model that names only real artifacts but formats them wrongly is
    indistinguishable from one that invents them. Those are different defects and must be
    reported separately.
    """
    return [tok.strip() for tok in s.split(",") if tok.strip()]


def lenient_egr(records: list[dict], allowed_lookup) -> dict:
    """EGR recomputed after splitting packed citations, isolating format violations."""
    tot = grounded = 0
    fmt_violations = 0
    for r in records:
        allowed = allowed_lookup(r)
        for c in r["cited"]:
            toks = _split_citation(c)
            if len(toks) > 1:
                fmt_violations += 1
            for t in toks:
                tot += 1
                if t in allowed:
                    grounded += 1
    n_cited = sum(len(r["cited"]) for r in records)
    return {
        "lenient_grounding_rate": grounded / max(1, tot),
        "format_violation_rate": fmt_violations / max(1, n_cited),
        "tokens_examined": tot,
    }


def egr(records: list[dict]) -> dict:
    """Evidence Grounding Rate over a set of explanations."""
    if not records:
        return {}
    per = np.array([r["egr"] for r in records])
    n_cited = np.array([len(r["cited"]) for r in records])
    n_hall = np.array([len(r["hallucinated"]) for r in records])
    return {
        "mean_egr": float(per.mean()),
        "median_egr": float(np.median(per)),
        "fully_grounded_rate": float((per == 1.0).mean()),
        "zero_grounded_rate": float((per == 0.0).mean()),
        "mean_citations": float(n_cited.mean()),
        "total_citations": int(n_cited.sum()),
        "total_hallucinated": int(n_hall.sum()),
        "hallucination_rate": float(n_hall.sum() / max(1, n_cited.sum())),
        # A citation longer than a plausible identifier is prose stuffed into the
        # evidence array - a distinct failure mode from naming a wrong artifact.
        "prose_stuffing_rate": float(np.mean([
            any(len(c) > 64 for c in r["cited"]) for r in records])),
    }


# ------------------------------------------------------------------------- CFN
def cfn(records: list[dict], df: pd.DataFrame, model) -> dict:
    """Counterfactual Necessity: ablate the cited artifacts and re-score."""
    if not records:
        return {}
    rows = [r["row_index"] for r in records]
    X = df.iloc[rows][K.FEATURES].to_numpy(dtype=np.float32)
    base_score = model.predict_proba(X)[:, 1]

    X_abl = X.copy()
    col_of = {c: j for j, c in enumerate(K.FEATURES)}
    n_ablated = []
    for i, r in enumerate(records):
        # Only grounded citations can be ablated; a hallucinated artifact has no column.
        hit = [col_of[c] for c in r["grounded"] if c in col_of]
        X_abl[i, hit] = 0.0
        n_ablated.append(len(hit))
    abl_score = model.predict_proba(X_abl)[:, 1]

    verdicts = np.array([r["llm_verdict"] for r in records])
    shift = abl_score - base_score
    # Signed shift toward the opposite of the asserted verdict. If the LLM said
    # MALWARE and the cited evidence carries that verdict, removing it should push
    # the score DOWN; if it said BENIGN, removing its evidence should push UP.
    toward_opposite = np.where(verdicts == 1, -shift, shift)
    flipped = (abl_score >= 0.5) != (base_score >= 0.5)

    # Control: ablate the same NUMBER of randomly chosen present artifacts, so the
    # measured effect is attributable to *which* artifacts were cited, not merely to
    # removing some features.
    rng = np.random.default_rng(C.SEED)
    X_rand = X.copy()
    bundle_idx = [col_of[c] for c in K.BUNDLE_FEATURES if c in col_of]
    for i, k in enumerate(n_ablated):
        present = [j for j in bundle_idx if X[i, j] > 0]
        if present and k:
            pick = rng.choice(present, size=min(k, len(present)), replace=False)
            X_rand[i, pick] = 0.0
    rand_score = model.predict_proba(X_rand)[:, 1]
    rand_shift = rand_score - base_score
    rand_toward = np.where(verdicts == 1, -rand_shift, rand_shift)

    return {
        "mean_abs_shift": float(np.abs(shift).mean()),
        "mean_shift_toward_opposite": float(toward_opposite.mean()),
        "flip_rate": float(flipped.mean()),
        "mean_artifacts_ablated": float(np.mean(n_ablated)),
        "random_control_mean_abs_shift": float(np.abs(rand_shift).mean()),
        "random_control_shift_toward_opposite": float(rand_toward.mean()),
        "random_control_flip_rate": float(
            ((rand_score >= 0.5) != (base_score >= 0.5)).mean()),
        # >1 means cited artifacts move the detector more than random present ones
        "necessity_lift": float(
            np.abs(shift).mean() / max(1e-9, np.abs(rand_shift).mean())),
    }


# -------------------------------------------------------------------------- AA
def attribution_agreement(records: list[dict], df: pd.DataFrame, model,
                          k: int = 10) -> dict:
    """Overlap of LLM citations with the SHAP top-k for the same sample."""
    if not records:
        return {}
    rows = [r["row_index"] for r in records]
    X = df.iloc[rows][K.FEATURES].to_numpy(dtype=np.float32)

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):          # older SHAP returns one array per class
        sv = sv[1]
    sv = np.asarray(sv)

    bundle_set = set(K.BUNDLE_FEATURES)
    feat = np.array(K.FEATURES)
    precisions, recalls, rand_precisions = [], [], []
    rng = np.random.default_rng(C.SEED)

    for i, r in enumerate(records):
        # Restrict the SHAP ranking to artifacts that were actually *available* to be
        # cited, otherwise the comparison is unfair: the LLM never saw SYS_nnn columns
        # or aggregate counters.
        mask = np.array([f in bundle_set and X[i, j] > 0
                         for j, f in enumerate(feat)])
        if mask.sum() == 0:
            continue
        idx = np.where(mask)[0]
        # Stable: SHAP values tie at exactly 0 for any artifact never split on, so an
        # unstable sort would make AA@k differ between machines on identical input.
        order = idx[np.argsort(-np.abs(sv[i, idx]), kind="stable")]
        topk = set(feat[order[:k]])

        cited = set(r["grounded"])
        if not cited:
            continue
        precisions.append(len(cited & topk) / len(cited))
        recalls.append(len(cited & topk) / max(1, len(topk)))

        # chance baseline: cite the same number of present artifacts at random
        pick = set(feat[rng.choice(idx, size=min(len(cited), len(idx)),
                                   replace=False)])
        rand_precisions.append(len(pick & topk) / len(pick))

    if not precisions:
        return {}
    return {
        "k": k,
        "n_evaluated": len(precisions),
        "aa_precision_at_k": float(np.mean(precisions)),
        "aa_recall_at_k": float(np.mean(recalls)),
        "random_baseline_precision": float(np.mean(rand_precisions)),
        "aa_lift_over_chance": float(
            np.mean(precisions) / max(1e-9, np.mean(rand_precisions))),
    }


# ------------------------------------------------------------------- abstention
def abstention_curve(records: list[dict]) -> list[dict]:
    """Coverage-risk trade-off when abstaining on poorly-grounded explanations."""
    if not records:
        return []
    e = np.array([r["egr"] for r in records])
    y = np.array([r["y_true"] for r in records])
    p = np.array([r["llm_verdict"] for r in records])
    out = []
    for tau in (0.0, 0.25, 0.5, 0.75, 1.0):
        keep = e >= tau
        if keep.sum() == 0:
            out.append({"egr_threshold": tau, "coverage": 0.0, "n_kept": 0})
            continue
        m = K.metrics(y[keep], p[keep])
        out.append({
            "egr_threshold": tau,
            "coverage": float(100 * keep.mean()),
            "n_kept": int(keep.sum()),
            "accuracy_on_kept": m["accuracy"],
            "f1_on_kept": m["f1"],
            "error_on_kept": 100 - m["accuracy"],
        })
    return out


# ------------------------------------------------------------------------- main
def evaluate(path) -> dict | None:
    d = json.loads(path.read_text())
    recs = d.get("records", [])
    if not recs:
        return None
    df = K.load(d["split"])
    # setdefault evaluates its default eagerly, so the previous form refitted XGBoost
    # once per result file and discarded it. Fit lazily.
    if "m" not in MODEL_CACHE:
        MODEL_CACHE["m"] = fit_base(*K.xy(K.load("train")))
    model = MODEL_CACHE["m"]

    # Reconstruct the closed set each explanation was shown, so lenient EGR can be
    # recomputed without re-querying the model.
    from llm_stage import build_bundle
    allowed_cache: dict[int, set] = {}

    def _allowed(rec):
        i = rec["row_index"]
        if i not in allowed_cache:
            allowed_cache[i] = build_bundle(df.iloc[i])[1]
        return allowed_cache[i]

    egr_stats = egr(recs)
    egr_stats.update(lenient_egr(recs, _allowed))

    return {
        "tag": path.stem.replace("llm_", ""),
        "split": d["split"], "model": d["model"], "mode": d["mode"],
        "budget": d["budget"], "n": len(recs),
        "verdict_metrics": d.get("llm_only_on_escalated", {}),
        "EGR": egr_stats,
        "CFN": cfn(recs, df, model),
        "AA": attribution_agreement(recs, df, model),
        "abstention": abstention_curve(recs),
    }


MODEL_CACHE: dict = {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="llm_*.json")
    a = ap.parse_args()

    print("=" * 96)
    print("FAITHDROID - Stage 5: automatic explanation-faithfulness audit")
    print("=" * 96)

    paths = sorted(C.RESULTS.glob(a.pattern))
    if not paths:
        print("no LLM result files found yet - run run_experiments.py first")
        return

    all_out = []
    for p in paths:
        r = evaluate(p)
        if r:
            all_out.append(r)

    print(f"\n{'config':<30s} {'n':>4s} {'EGRs':>6s} {'EGRl':>6s} {'fmt%':>6s} "
          f"{'full':>5s} {'hall':>6s} {'CFNx':>5s} {'flip':>6s} {'AA@10':>6s} {'AAx':>5s}")
    print("-" * 106)
    for r in all_out:
        E, F, A = r["EGR"], r["CFN"], r["AA"]
        print(f"{r['tag']:<30s} {r['n']:4d} "
              f"{E.get('mean_egr', 0):6.3f} "
              f"{E.get('lenient_grounding_rate', 0):6.3f} "
              f"{100 * E.get('format_violation_rate', 0):5.1f}% "
              f"{100 * E.get('fully_grounded_rate', 0):4.0f}% "
              f"{100 * E.get('hallucination_rate', 0):5.1f}% "
              f"{F.get('necessity_lift', 0):5.2f} "
              f"{100 * F.get('flip_rate', 0):5.1f}% "
              f"{A.get('aa_precision_at_k', 0):6.3f} "
              f"{A.get('aa_lift_over_chance', 0):5.2f}")

    K.save_json(all_out, C.RESULTS / "faithfulness.json")


if __name__ == "__main__":
    main()
