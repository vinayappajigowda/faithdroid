"""End-to-end demonstration on a single application.

Runs one app through the whole cascade and prints what the system actually did at each
stage: what the cheap detector thought, whether the gate escalated, what the LLM said and
which artifacts it cited, and how that explanation scores under the faithfulness audit.

This exists to be shown. The point of the thesis is that an analyst can see the evidence
behind a verdict and that the evidence can be checked, so the system should be
demonstrable on one app in a few seconds rather than only as a table of aggregates.

Examples
--------
    python3 code/demo.py                          # a malware sample the cascade fixes
    python3 code/demo.py --split Test3 --index 7  # a specific application
    python3 code/demo.py --find-disagreement      # a case where LLM and detector differ
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

import common as K
import config as C
from cascade import BAND_EDGES, fit_override_bands, in_bands
from conformal import fit_base
from faithfulness import _split_citation
from llm_stage import CONTEXT_INSTRUCTION, build_bundle, query
from retrieval import NeighbourIndex


def _console_supports(ch: str) -> bool:
    enc = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        ch.encode(enc)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


# A Windows console defaults to cp1252 and cannot encode U+2500, which used to abort this
# script on its first printed line. Prefer UTF-8, fall back to ASCII.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError, ValueError):
    pass

RULE_CHAR = "─" if _console_supports("─") else "-"

if sys.stdout.isatty():
    BOLD, DIM, OFF = "\033[1m", "\033[2m", "\033[0m"
    RED, GRN, YEL, CYN = "\033[31m", "\033[32m", "\033[33m", "\033[36m"
else:
    BOLD = DIM = OFF = RED = GRN = YEL = CYN = ""


def rule(title: str = "") -> None:
    print(f"\n{BOLD}{RULE_CHAR * 78}{OFF}")
    if title:
        print(f"{BOLD}{title}{OFF}")
        print(f"{BOLD}{RULE_CHAR * 78}{OFF}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="Test3")
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--find-disagreement", action="store_true",
                    help="pick a case where the LLM and the detector disagree")
    ap.add_argument("--model", default=C.LLM_PRIMARY)
    a = ap.parse_args()

    df = K.load(a.split)
    df.attrs["split"] = a.split
    pv = json.loads((C.RESULTS / "conformal_pvalues.json").read_text())[a.split]
    proba = np.asarray(pv["proba"])
    y = np.asarray(pv["y"])

    # ---- choose an application to demonstrate
    if a.index is not None:
        i = a.index
    else:
        cached = C.RESULTS / f"llm_main_{a.split}_b20.json"
        cached = cached if cached.exists() else C.RESULTS / f"llm_main_{a.split}.json"
        pick = None
        if cached.exists():
            recs = json.loads(cached.read_text())["records"]
            base_pred = np.asarray(pv["pred"])
            for r in recs:
                j = r["row_index"]
                llm_right = r["llm_verdict"] == r["y_true"]
                base_right = base_pred[j] == r["y_true"]
                if a.find_disagreement and r["llm_verdict"] != base_pred[j]:
                    pick = j
                    break
                if not a.find_disagreement and llm_right and not base_right:
                    pick = j
                    break
        i = pick if pick is not None else int(np.argmin(np.abs(proba - 0.5)))

    row = df.iloc[i]
    truth = "MALWARE" if row["Malware"] == 1 else "BENIGN"
    fam = str(row.get("MalFamily", "")) or "-"

    rule(f"FAITHDROID demonstration - {a.split} application #{i}")
    print(f"  package     {str(row.get('Package', 'unknown'))[:58]}")
    print(f"  sha256      {str(row.get('sha256', ''))[:48]}...")
    print(f"  first seen  {row['ts'].date()}")
    print(f"  ground truth{'':1s}{RED if truth=='MALWARE' else GRN}{truth}{OFF}"
          f"   family: {fam}")

    # ---- stage 2
    rule("Stage 2  cheap detector (XGBoost over 475 hybrid features)")
    p = float(proba[i])
    base_verdict = "MALWARE" if p >= 0.5 else "BENIGN"
    ok = "correct" if base_verdict == truth else f"{RED}WRONG{OFF}"
    print(f"  P(malware) = {p:.4f}   ->  verdict {base_verdict}   [{ok}]")

    # ---- gate
    rule("Gate  should this be escalated?")
    score = (1.0 - abs(p - 0.5) * 2.0) * (0.5 + p)
    all_scores = (1.0 - np.abs(proba - 0.5) * 2.0) * (0.5 + proba)
    pct = 100 * (all_scores > score).mean()
    print(f"  uncertainty score = {score:.4f}   (top {pct:.1f}% most uncertain)")
    escalated = pct <= 20.0
    print(f"  -> {'ESCALATE to LLM' if escalated else 'retain base verdict, no LLM cost'}")
    if not escalated:
        print(f"  {DIM}(this app would not normally reach the LLM; "
              f"continuing anyway for the demonstration){OFF}")

    # ---- stage 3
    rule("Stage 3  LLM reasoning over a CLOSED evidence bundle")
    bundle, allowed = build_bundle(row)
    idx = NeighbourIndex(k=6)
    nbs = idx.query(row)
    print(f"  closed citation vocabulary: {len(allowed)} artifacts")
    print(f"  retrieved neighbours (all pre-2015): {idx.summarise(nbs)}")
    for nb in nbs[:3]:
        print(f"    {DIM}sim={nb['similarity']:.2f}  {nb['label']:<8s}"
              f"{(' ' + nb['family']) if nb['family'] else ''}{OFF}")

    prompt = CONTEXT_INSTRUCTION.format(
        bundle=bundle, prevalence=100 * C.TARGET_PREVALENCE,
        neighbours=idx.render(nbs), neighbour_summary=idx.summarise(nbs))
    import hashlib
    key = hashlib.sha256((str(row.get("sha256", i)) + prompt).encode()).hexdigest()[:24]
    print(f"  querying {a.model} ...", end="", flush=True)
    res = query(prompt, a.model, "context", key)
    if res is None:
        print(f" {RED}failed{OFF}")
        print(f"\n  Not in the cache, so it needed an Ollama server at {C.OLLAMA_URL}\n"
              f"  serving '{a.model}':  ollama serve &  ollama pull {a.model}\n"
              f"  If Ollama IS running, the cache key missed because the environment does\n"
              f"  not match requirements.txt. Run: python3 code/envcheck.py\n")
        raise SystemExit(1)
    took = "cached" if res.get("cached") else f"{res.get('latency_s', 0):.1f}s"
    print(f" done ({took})")

    llm_verdict = res["verdict"]
    ok = "correct" if llm_verdict == truth else f"{RED}WRONG{OFF}"
    print(f"\n  verdict: {llm_verdict}   [{ok}]")
    print(f"  reasoning: {res['reasoning'][:300]}")

    # ---- stage 4
    rule("Stage 4  automatic faithfulness audit")
    cited = res["cited_evidence"]
    grounded = [c for c in cited if c in allowed]
    print(f"  cited evidence ({len(cited)} items):")
    for c in cited:
        mark = f"{GRN}OK    {OFF}" if c in allowed else f"{RED}UNGROUNDED{OFF}"
        toks = _split_citation(c)
        note = ""
        if c not in allowed and len(toks) > 1 and all(t in allowed for t in toks):
            note = f"  {YEL}(format violation: {len(toks)} artifacts in one field){OFF}"
        print(f"    {mark}  {c[:56]}{note}")
    egr = len(grounded) / len(cited) if cited else 0.0
    print(f"\n  {BOLD}EGR = {egr:.3f}{OFF}  "
          f"({len(grounded)}/{len(cited)} citations verified against the app itself,"
          f" no human needed)")

    # counterfactual necessity for this one app
    model = fit_base(*K.xy(K.load("train")))
    X = df.iloc[[i]][K.FEATURES].to_numpy(dtype=np.float32)
    before = float(model.predict_proba(X)[:, 1][0])
    col = {c: j for j, c in enumerate(K.FEATURES)}
    Xa = X.copy()
    for c in grounded:
        if c in col:
            Xa[0, col[c]] = 0.0
    after = float(model.predict_proba(Xa)[:, 1][0])
    print(f"  CFN: ablating the cited artifacts moves P(malware) "
          f"{before:.4f} -> {after:.4f}  (shift {abs(after-before):.4f})")

    # ---- final decision
    rule("Final decision")
    bands = fit_override_bands()
    in_band = bool(in_bands(np.array([p]), bands)[0])
    print(f"  override band {bands}: detector probability {p:.3f} is "
          f"{'INSIDE' if in_band else 'outside'}")
    if escalated and in_band:
        final = llm_verdict
        why = "LLM verdict adopted (detector unreliable in this band)"
    else:
        final = base_verdict
        why = ("detector verdict kept (outside the unreliable band)" if escalated
               else "detector verdict kept (not escalated)")
    colour = RED if final == "MALWARE" else GRN
    mark = f"{GRN}correct{OFF}" if final == truth else f"{RED}wrong{OFF}"
    print(f"\n  {BOLD}FINAL: {colour}{final}{OFF}{BOLD}   [{mark}]{OFF}   {DIM}{why}{OFF}")
    print(f"  ground truth was {truth}\n")


if __name__ == "__main__":
    main()
