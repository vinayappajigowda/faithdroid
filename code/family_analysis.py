"""Stage 8 - per-family breakdown of detection and escalation.

LAMD reports family-level classification quality and finds that Adware, PUA and Riskware
are hardest, because their behaviour is least distinct from benign software. KronoDroid
carries 240 family labels, so the same question can be asked here: which families does
the cheap detector miss, and does escalation recover them?

This matters beyond curiosity. If escalation only helps on families the detector already
handles, the cascade is redundant; if it recovers families the detector systematically
misses, it is addressing a real blind spot.

Run:  python3 code/family_analysis.py
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

import common as K
import config as C

MIN_SAMPLES = 5


def main() -> None:
    print("=" * 92)
    print("FAITHDROID - Stage 8: per-family analysis")
    print("=" * 92)

    pv = json.loads((C.RESULTS / "conformal_pvalues.json").read_text())

    # ---------------------------------------------------- family prevalence over time
    print("\n[1] Malware families by evaluation window (top 8 per window)")
    for split in K.TEST_SPLITS:
        df = K.load(split)
        mal = df[df.Malware == 1]
        vc = mal["MalFamily"].astype(str).value_counts()
        top = ", ".join(f"{f}({n})" for f, n in vc.head(8).items())
        print(f"  {split}: {len(vc)} families among {len(mal)} malware")
        print(f"    {top}")

    # ------------------------------------------------- drift in family composition
    print("\n[2] Family turnover between windows")
    fams = {}
    for split in K.TEST_SPLITS:
        df = K.load(split)
        fams[split] = set(df[df.Malware == 1]["MalFamily"].astype(str))
    a, b, c = (fams[s] for s in K.TEST_SPLITS)
    print(f"  Test1 families: {len(a)}   Test2: {len(b)}   Test3: {len(c)}")
    print(f"  in Test3 but never seen in Test1: {len(c - a)} "
          f"({100 * len(c - a) / max(1, len(c)):.0f}% of Test3 families)")
    print(f"  present in all three windows: {len(a & b & c)}")
    print("  -> family turnover is the concrete form the concept drift takes here")

    # ----------------------------------------- detector recall by family, per window
    print("\n[3] Base-detector recall by family "
          f"(families with >= {MIN_SAMPLES} samples in a window)")
    print(f"  {'family':<22s} {'window':<7s} {'n':>4s} {'recall':>8s}")
    weak = []
    for split in K.TEST_SPLITS:
        df = K.load(split)
        pred = np.asarray(pv[split]["pred"])
        mal_mask = (df.Malware == 1).to_numpy()
        famv = df["MalFamily"].astype(str).to_numpy()
        agg = defaultdict(lambda: [0, 0])
        for j in np.where(mal_mask)[0]:
            agg[famv[j]][0] += 1
            agg[famv[j]][1] += int(pred[j] == 1)
        for f, (n, hit) in sorted(agg.items(), key=lambda kv: kv[1][0], reverse=True):
            if n < MIN_SAMPLES:
                continue
            rec = hit / n
            flag = "  <- missed entirely" if rec == 0 else ""
            print(f"  {f:<22s} {split:<7s} {n:4d} {100 * rec:7.1f}%{flag}")
            if rec < 0.5:
                weak.append((f, split, n, rec))

    # --------------------------------------- does escalation reach the weak families?
    print("\n[4] Does escalation reach the families the detector misses?")
    for tag, split in (("main_Test1", "Test1"), ("main_Test2", "Test2"),
                       ("main_Test3_b20", "Test3")):
        p = C.RESULTS / f"llm_{tag}.json"
        if not p.exists():
            continue
        recs = json.loads(p.read_text())["records"]
        df = K.load(split)
        pred = np.asarray(pv[split]["pred"])
        esc = {r["row_index"]: r for r in recs}
        famv = df["MalFamily"].astype(str).to_numpy()
        mal_idx = np.where((df.Malware == 1).to_numpy())[0]

        missed = [j for j in mal_idx if pred[j] == 0]
        missed_esc = [j for j in missed if j in esc]
        recovered = [j for j in missed_esc if esc[j]["llm_verdict"] == 1]
        print(f"  {split}: detector missed {len(missed)} malware; "
              f"{len(missed_esc)} were escalated "
              f"({100 * len(missed_esc) / max(1, len(missed)):.0f}%); "
              f"LLM recovered {len(recovered)} "
              f"({100 * len(recovered) / max(1, len(missed_esc)):.0f}% of those)")
        if recovered:
            rf = defaultdict(int)
            for j in recovered:
                rf[famv[j]] += 1
            top = ", ".join(f"{f}({n})" for f, n in
                            sorted(rf.items(), key=lambda kv: -kv[1])[:6])
            print(f"    families recovered: {top}")

    K.save_json({"weak_families": [
        {"family": f, "split": s, "n": n, "recall": r} for f, s, n, r in weak]},
        C.RESULTS / "family_analysis.json")


if __name__ == "__main__":
    main()
