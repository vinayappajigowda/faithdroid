"""Stage 8 - adversarial evaluation: can a grounded explanation be wrong?

This is the security question the rest of the evaluation does not answer.
\\citet{saqib2024comprehensive} warn that explainable malware detectors are *more* exposed to
adversarial attack, and \\citet{nadeem2023sok} note that malware authors can evade detection
while masking the very features used for evasion. A faithfulness metric that can be satisfied
by an evasive sample would be worse than no metric at all, because a well-grounded explanation
for a wrong verdict carries unearned credibility.

Threat model
------------
Feature ADDITION only. An attacker can declare extra permissions, register extra components,
or invoke extra system calls without breaking the application; removing artifacts the malware
genuinely needs would break its function. This is the standard constraint in the Android
evasion literature and it is what makes the attack realistic rather than an arbitrary
perturbation of a feature vector.

The attacker is assumed to know the detector (white-box, via SHAP) - the strongest reasonable
assumption, so the result is an upper bound on evasion.

Three questions
---------------
Q1  Can the cheap detector be evaded by adding benign-looking artifacts, and how many?
Q2  When it is evaded, does the LLM stage catch the evasion or follow it?
Q3  For samples where BOTH are fooled, does the explanation remain well grounded? If EGR stays
    high on a wrong verdict, the metric measures grounding but not correctness, and that
    limitation must be stated explicitly rather than left for a reader to discover.

Run:  python3 code/adversarial.py
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import shap

import common as K
import config as C
from conformal import fit_base

# Attack budgets: how many benign-indicative artifacts the attacker adds.
BUDGETS = [1, 2, 5, 10, 20, 40]


def benign_direction(model, X_target: np.ndarray, top_n: int = 80) -> list[int]:
    """Artifacts whose ADDITION most reduces the detector's malware score.

    A first attempt ranked candidates by mean SHAP value over the training set. That was
    wrong and the experiment caught it: the mean is dominated by rows where the feature is
    absent, so the ranking bore no relation to the effect of *adding* it. The resulting
    "benign-indicative" list was headed by SEND_SMS and READ_PHONE_STATE - malware
    indicators - and adding them raised the malware score from 0.967 to 0.994.

    The effect of an addition is therefore measured directly: add each candidate to every
    target sample that lacks it and record the mean change in predicted probability. This
    is a greedy white-box attack and needs no proxy.
    """
    base = model.predict_proba(X_target)[:, 1]
    bundle = set(K.BUNDLE_FEATURES)
    cand_idx = [j for j, f in enumerate(K.FEATURES) if f in bundle]

    effects = []
    for j in cand_idx:
        absent = X_target[:, j] == 0
        if absent.sum() < 5:          # already present nearly everywhere: useless to add
            continue
        Xa = X_target.copy()
        Xa[:, j] = np.maximum(Xa[:, j], 1.0)
        delta = (model.predict_proba(Xa)[:, 1] - base)[absent].mean()
        effects.append((j, float(delta)))

    effects.sort(key=lambda t: t[1])   # most negative delta first
    return [j for j, _ in effects[:top_n]]


def attack(X: np.ndarray, cols: list[int], k: int) -> np.ndarray:
    """Add the k highest-value artifacts to every row (presence only, never removal)."""
    Xa = X.copy()
    for j in cols[:k]:
        Xa[:, j] = np.maximum(Xa[:, j], 1.0)
    return Xa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="Test1")
    ap.add_argument("--llm-budget", type=int, default=60,
                    help="how many evaded samples to re-run through the LLM")
    a = ap.parse_args()

    print("=" * 92)
    print("FAITHDROID - Stage 8: adversarial evaluation")
    print("=" * 92)

    Xtr, ytr = K.xy(K.load("train"))
    model = fit_base(Xtr, ytr)
    df = K.load(a.split)
    X, y = K.xy(df)

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    # Attack only malware the detector currently catches - evading an already-missed
    # sample is not an attack.
    caught = np.where((y == 1) & (pred == 1))[0]
    print(f"\n{a.split}: {len(caught)} malware samples currently detected "
          f"(of {int(y.sum())} total malware)")

    Xc = X[caught]
    print('\n  measuring the true effect of adding each candidate artifact...')
    cols = benign_direction(model, Xc)
    names = [K.FEATURES[j] for j in cols[:10]]
    print(f"\ntop benign-indicative artifacts an attacker would add:")
    print(f"  {', '.join(names)}")

    print(f"\n[Q1] detector evasion by attack budget")
    print(f"  {'artifacts added':>16s} {'evaded':>8s} {'evasion rate':>13s} "
          f"{'mean P(mal)':>12s}")
    results = {"split": a.split, "n_attacked": len(caught), "budgets": {}}
    evaded_at = {}
    for k in BUDGETS:
        Xa = attack(Xc, cols, k)
        pa = model.predict_proba(Xa)[:, 1]
        ev = pa < 0.5
        evaded_at[k] = ev
        results["budgets"][str(k)] = {
            "evaded": int(ev.sum()), "n": len(ev),
            "evasion_rate": float(ev.mean()),
            "mean_proba": float(pa.mean()),
        }
        print(f"  {k:16d} {int(ev.sum()):8d} {100 * ev.mean():12.1f}% {pa.mean():12.3f}")

    # ---------------------------------------------------------------- Q2 and Q3
    # Take the smallest budget that evades a useful number of samples.
    k_use = next((k for k in BUDGETS if evaded_at[k].sum() >= 20), BUDGETS[-1])
    ev = evaded_at[k_use]
    ev_idx = caught[ev][: a.llm_budget]
    print(f"\n[Q2/Q3] re-running {len(ev_idx)} evaded samples through the LLM "
          f"(attack budget k={k_use})")
    if len(ev_idx) == 0:
        print("  no evaded samples to test")
        K.save_json(results, C.RESULTS / f"adversarial_{a.split}.json")
        return

    from llm_stage import (CONTEXT_INSTRUCTION, build_bundle, query)
    from retrieval import NeighbourIndex
    index = NeighbourIndex(k=6)

    # The attack is applied per-row below, to the row that builds the evidence bundle.
    col_names = {j: K.FEATURES[j] for j in cols[:k_use]}
    recs = []
    for n, i in enumerate(ev_idx, 1):
        row = df.iloc[i].copy()
        # Reflect the attack in the row so the bundle the LLM sees is the attacked app.
        for j, nm in col_names.items():
            row[nm] = max(float(row.get(nm, 0)), 1.0)
        bundle, allowed = build_bundle(row)
        nbs = index.query(row)
        prompt = CONTEXT_INSTRUCTION.format(
            bundle=bundle, prevalence=100 * C.TARGET_PREVALENCE,
            neighbours=index.render(nbs), neighbour_summary=index.summarise(nbs))
        key = hashlib.sha256(("adv" + str(row.get("sha256", i)) + prompt).encode()
                             ).hexdigest()[:24]
        res = query(prompt, C.LLM_PRIMARY, "adversarial", key)
        if res is None:
            continue
        cited = res["cited_evidence"]
        grounded = [c for c in cited if c in allowed]
        # Did the attacker's injected artifacts end up being cited as evidence?
        injected = set(col_names.values())
        recs.append({
            "row_index": int(i),
            "y_true": 1,
            "llm_verdict": 1 if res["verdict"] == "MALWARE" else 0,
            "cited": cited, "grounded": grounded,
            "egr": len(grounded) / len(cited) if cited else 0.0,
            "cited_injected": [c for c in grounded if c in injected],
            "reasoning": res["reasoning"],
        })
        if n % 20 == 0 or n == len(ev_idx):
            print(f"    {n}/{len(ev_idx)}")

    # If the reasoning stage produced nothing - Ollama unreachable, or the server dropped
    # part way through - every aggregate below is a mean over an empty array, i.e. NaN,
    # and writing that out would overwrite a good results file with an unusable one. Keep
    # the detector-evasion measurements, which are real, and stop before the LLM section.
    if not recs:
        print("\n  ! no LLM responses were obtained, so Q2/Q3 cannot be evaluated.")
        print(f"    Is Ollama serving {C.LLM_PRIMARY} at {C.OLLAMA_URL}?")
        out = C.RESULTS / f"adversarial_{a.split}.json"
        if out.exists():
            print(f"    Leaving the existing {out.name} untouched rather than "
                  "replacing it with empty results.")
            raise SystemExit(1)
        K.save_json(results, out)
        raise SystemExit(1)

    llm = np.array([r["llm_verdict"] for r in recs])
    egr = np.array([r["egr"] for r in recs])
    both_fooled = llm == 0            # detector already evaded by construction
    n_inj_cited = sum(len(r["cited_injected"]) > 0 for r in recs)

    print(f"\n  detector evaded : {len(recs)}/{len(recs)} (by construction)")
    print(f"  LLM also fooled : {int(both_fooled.sum())}/{len(recs)} "
          f"({100 * both_fooled.mean():.1f}%)")
    print(f"  LLM caught it   : {int((~both_fooled).sum())}/{len(recs)} "
          f"({100 * (~both_fooled).mean():.1f}%)")
    print(f"\n  mean EGR overall            : {egr.mean():.3f}")
    if both_fooled.sum():
        print(f"  mean EGR when BOTH fooled   : {egr[both_fooled].mean():.3f}   "
              f"<- grounded explanation for a WRONG verdict")
    if (~both_fooled).sum():
        print(f"  mean EGR when LLM caught it : {egr[~both_fooled].mean():.3f}")
    print(f"  explanations citing an injected artifact: {n_inj_cited}/{len(recs)} "
          f"({100 * n_inj_cited / max(1, len(recs)):.1f}%)")

    results["llm"] = {
        "attack_budget": k_use, "n_tested": len(recs),
        "llm_fooled": int(both_fooled.sum()),
        "llm_fooled_rate": float(both_fooled.mean()),
        "llm_caught": int((~both_fooled).sum()),
        "mean_egr": float(egr.mean()),
        "mean_egr_when_fooled": float(egr[both_fooled].mean()) if both_fooled.sum() else None,
        "mean_egr_when_caught": float(egr[~both_fooled].mean()) if (~both_fooled).sum() else None,
        "explanations_citing_injected": n_inj_cited,
        "records": recs,
    }
    K.save_json(results, C.RESULTS / f"adversarial_{a.split}.json")


if __name__ == "__main__":
    main()
