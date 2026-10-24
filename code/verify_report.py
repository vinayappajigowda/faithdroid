"""Cross-check every hand-written number in the report against the saved results.

Generated tables cannot drift from the experiments, but prose numbers are typed by hand and
can. This script asserts the claims made in the .tex files against the JSON, so a stale
figure in the narrative fails loudly instead of shipping.

Run:  python3 code/verify_report.py
"""
from __future__ import annotations

import json
import re
import sys

import numpy as np

import config as C

TEXT = C.RESULTS.parent / "report" / "text"
FAILS: list[str] = []
CHECKS = 0


def load(name):
    p = C.RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def check(label: str, claimed, actual, tol=0.02):
    """Assert a claimed number matches the computed one."""
    global CHECKS
    CHECKS += 1
    if actual is None:
        FAILS.append(f"{label}: no data to verify against (claimed {claimed})")
        return
    if abs(claimed - actual) > tol:
        FAILS.append(f"{label}: report says {claimed}, results say {actual:.4f}")


def body() -> str:
    return "\n".join(p.read_text() for p in sorted(TEXT.glob("*.tex"))
                     if p.name not in ("generated_tables.tex", "generated_macros.tex"))


def macro_checks() -> None:
    """The prose states its numbers through generated macros; check they resolve.

    Prose numbers used to be typed by hand, which is why this script carried a duplicate
    of every claim. They are now emitted by make_macros.py from the saved results, so the
    narrative updates itself on a rerun. What remains to verify is that every macro the
    prose uses exists, and that the generated file is not stale.
    """
    gen = TEXT / "generated_macros.tex"
    if not gen.exists():
        FAILS.append("generated_macros.tex is missing - run code/make_macros.py")
        return
    gen_text = gen.read_text()
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", gen_text))

    # A LaTeX control sequence may contain letters only - a digit terminates the name, so
    # \newcommand{\AdvEvasionTestOneK1} is not definable and fails the build. This caught
    # exactly that on twelve macros, so it stays.
    illegal = sorted(n for n in defined if not n.isalpha())
    if illegal:
        FAILS.append(f"macro names are not pure letters, LaTeX cannot define them: {illegal}")
    unbalanced = sorted(n for n, v in re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", gen_text)
                        if v.count("{") != v.count("}"))
    if unbalanced:
        FAILS.append(f"macro expansions have unbalanced braces: {unbalanced}")

    used: set[str] = set()
    for p in sorted(TEXT.glob("*.tex")):
        if p.name in ("generated_macros.tex", "generated_tables.tex"):
            continue
        used |= set(re.findall(r"\\([A-Za-z]+)\{\}", p.read_text()))
    undefined = used - defined
    if undefined:
        FAILS.append(f"prose uses undefined macros: {sorted(undefined)}")

    # Stale check: regenerating must reproduce the committed file exactly.
    import make_macros
    if make_macros.render(make_macros.collect()) != gen.read_text():
        FAILS.append("generated_macros.tex is stale - rerun code/make_macros.py")
    print(f"  info: {len(used)} generated macros used in prose, "
          f"{len(defined)} defined")


def claim_checks() -> None:
    """The qualitative claims the thesis actually makes.

    These are not numbers but directions, and they must hold whatever the rerun produces.
    If one of them flips, the narrative is wrong no matter how fresh the figures are.
    """
    st = load("statistics.json")
    if st and "Test2" in st:
        if st["Test2"]["mcnemar"]["p_value"] < 0.05:
            FAILS.append("Test2 is now SIGNIFICANT - the report says it is not")
        if not (st["Test2"]["ci95"][0] < 0 < st["Test2"]["ci95"][1]):
            FAILS.append("Test2 CI no longer includes zero - report claim is stale")

    ceil = load("ceiling.json")
    if ceil:
        for split in ("Test1", "Test2", "Test3"):
            if split not in ceil:
                continue
            if ceil[split]["mcnemar"]["p_value"] < 0.05:
                FAILS.append(f"ceiling {split} is now significant - report says it is not")
            if ceil[split]["delta_llm_minus_base"] > 0:
                FAILS.append(f"ceiling {split} now favours the LLM - report says otherwise")

    for split in ("Test1", "Test3"):
        adv = load(f"adversarial_{split}.json")
        L = adv.get("llm") if adv else None
        if not L:
            continue
        caught = L.get("mean_egr_when_caught")
        if caught is not None and L["mean_egr_when_fooled"] < caught - 0.001:
            FAILS.append(
                f"adv {split}: EGR is now LOWER when fooled - the report claims otherwise")


def structural_checks(txt: str) -> None:
    """Dangling cross-references and unknown citation keys."""
    gen = TEXT / "generated_tables.tex"
    labels = set(re.findall(r"\\label\{([^}]+)\}", txt))
    if gen.exists():
        labels |= set(re.findall(r"\\label\{([^}]+)\}", gen.read_text()))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", txt))
    missing = refs - labels
    if missing:
        FAILS.append(f"dangling \\ref targets: {sorted(missing)}")

    bib = (C.RESULTS.parent / "report" / "refs.bib").read_text()
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited: set[str] = set()
    for m in re.findall(r"\\cite[a-z]*\{([^}]+)\}", txt):
        cited |= {k.strip() for k in m.split(",")}
    unknown = cited - keys
    if unknown:
        FAILS.append(f"cited but not in refs.bib: {sorted(unknown)}")
    unused = keys - cited
    if unused:
        print(f"  info: in refs.bib but never cited: {sorted(unused)}")


def report() -> None:
    print(f"\n{CHECKS} numeric checks run against saved results")
    if FAILS:
        print(f"\n{len(FAILS)} PROBLEM(S):")
        for f in FAILS:
            print(f"  - {f}")
        raise SystemExit(1)
    print("all report numbers match the experiment output")


def main() -> None:
    txt = body()
    macro_checks()
    claim_checks()

    # The value assertions below duplicate numbers that the prose now takes from
    # generated_macros.tex, so they cannot catch a stale narrative any more. They are
    # kept as an opt-in regression check against the shipped results: after a rerun the
    # numbers legitimately change, and `--values` is the flag that says "these are still
    # the figures I expect".
    if "--values" not in sys.argv:
        structural_checks(txt)
        report()
        return

    # ---------------------------------------------------------------- baselines
    b = load("baselines.json")
    if b:
        check("RF Test3 F1 (17.34)", 17.34, b["random_forest"]["Test3"]["f1"])
        check("RF Test2 F1 (76.98)", 76.98, b["random_forest"]["Test2"]["f1"])
        check("RF Test3 recall (9.82)", 9.82, b["random_forest"]["Test3"]["recall"])
        check("XGB Test2 F1 (81.56)", 81.56, b["xgboost"]["Test2"]["f1"])
        check("XGB Test3 F1 (56.23)", 56.23, b["xgboost"]["Test3"]["f1"])
        check("Drebin Test1 precision (34.91)", 34.91,
              b["drebin_svm"]["Test1"]["precision"])
        check("Drebin Test2 precision (39.67)", 39.67,
              b["drebin_svm"]["Test2"]["precision"])
        check("Drebin Test1 FPR (16.94)", 16.94, b["drebin_svm"]["Test1"]["fpr"])
        check("Drebin Test3 FPR (14.79)", 14.79, b["drebin_svm"]["Test3"]["fpr"])

    # ------------------------------------------------------------------ routing
    r = load("routing_diagnostic.json")
    if r:
        def mean_lift(sig, bud):
            return float(np.mean(r[sig][bud])) if sig in r and bud in r[sig] else None
        check("margin_x_malware @10% (13.11)", 13.11,
              mean_lift("margin_x_malware", "0.1"), tol=0.05)
        check("margin_x_malware @20% (14.42)", 14.42,
              mean_lift("margin_x_malware", "0.2"), tol=0.05)
        check("prob_margin @10% (11.13)", 11.13,
              mean_lift("prob_margin", "0.1"), tol=0.05)
        check("prob_malware @20% (20.23)", 20.23,
              mean_lift("prob_malware", "0.2"), tol=0.05)
        check("prob_malware @5% (1.82)", 1.82,
              mean_lift("prob_malware", "0.05"), tol=0.05)
        check("conformal_ambiguity @10% (0.49)", 0.49,
              mean_lift("conformal_ambiguity", "0.1"), tol=0.05)

    # ---------------------------------------------------- split sizes in prose
    meta = json.loads((C.PROC / "meta.json").read_text())
    ss = meta["split_sizes"]
    for name, n in (("train", 23581), ("calib", 10107), ("Test1", 1627),
                    ("Test2", 1591), ("Test3", 2848)):
        check(f"split size {name} ({n})", n, ss[name]["n"], tol=0)
    for name, m in (("Test1", 163), ("Test2", 159), ("Test3", 285)):
        check(f"malware count {name} ({m})", m, ss[name]["malware"], tol=0)

    # feature-group counts quoted in methodology / design
    fg = meta["feature_groups"]
    check("dynamic feature count (288)", 288, fg["dynamic"], tol=0)
    check("named syscalls (218)", 218, fg["dynamic_named"], tol=0)
    check("unnamed SYS_nnn (70)", 70, fg["dynamic_unnamed"], tol=0)
    check("static permissions (166)", 166, fg["static"], tol=0)
    total = fg["dynamic"] + fg["static"] + fg["aggregate"]
    check("total features (475)", 475, total, tol=0)

    # --------------------------------------------------------------- cascade prose
    casc = load("cascade.json")
    if casc:
        by_tag = {c["tag"]: c for c in casc}

        # Claims made in evaluation.tex about Test1 and Test2.
        if "main_Test1" in by_tag:
            m = by_tag["main_Test1"]
            check("Test1 base@escalated (50.9)", 50.9,
                  m["base_on_escalated"]["accuracy"], tol=0.1)
            check("Test1 LLM@escalated (76.7)", 76.7,
                  m["llm_on_escalated"]["accuracy"], tol=0.1)
            check("Test1 cascade F1 (75.88)", 75.88, m["cascade"]["f1"], tol=0.05)
            check("Test1 base F1 (66.12)", 66.12, m["base_only"]["f1"], tol=0.05)
        if "main_Test2" in by_tag:
            m = by_tag["main_Test2"]
            check("Test2 base@escalated (74.8)", 74.8,
                  m["base_on_escalated"]["accuracy"], tol=0.1)
            check("Test2 LLM@escalated (78.6)", 78.6,
                  m["llm_on_escalated"]["accuracy"], tol=0.1)
            check("Test2 cascade F1 (82.76)", 82.76, m["cascade"]["f1"], tol=0.05)
            check("Test2 gain (1.19)", 1.19, m["delta_f1_vs_base"], tol=0.05)

        # Blind-override degradation on Test3 quoted in the second negative result.
        for tag, claimed in (("main_Test3_b20@b2", 2.49), ("main_Test3_b20@b5", -0.80),
                             ("main_Test3_b20@b10", -8.65), ("main_Test3_b20@b15", -12.93),
                             ("main_Test3_b20", -15.56)):
            if tag in by_tag:
                check(f"blind dF1 {tag} ({claimed})", claimed,
                      by_tag[tag]["delta_f1_vs_base"], tol=0.05)

        # Banded-override recovery.
        for tag, claimed in (("main_Test3_b20", 2.30), ("main_Test1", 6.51),
                             ("main_Test2", 1.53)):
            if tag in by_tag:
                check(f"banded dF1 {tag} ({claimed})", claimed,
                      by_tag[tag]["delta_f1_banded_vs_base"], tol=0.05)
        if "main_Test3_b20" in by_tag:
            bd = by_tag["main_Test3_b20"]["banded_detail"]
            check("Test3 override rate (3.4)", 3.4, bd["override_rate"], tol=0.1)
            check("Test3 base@esc at b20 (68.9)", 68.9,
                  by_tag["main_Test3_b20"]["base_on_escalated"]["accuracy"], tol=0.1)
            check("Test3 blind FNR at b20 (65.96)", 65.96,
                  by_tag["main_Test3_b20"]["cascade"]["fnr"], tol=0.05)

        # The 888-token prompt figure quoted in the cost caveat.
        ctx = [c for c in casc if c["mode"] == "context" and c["n_escalated"] > 50]
        if ctx:
            avg_prompt = np.mean([c["cost"]["prompt_tokens"] / c["n_escalated"]
                                  for c in ctx])
            check("mean prompt tokens (888)", 888, avg_prompt, tol=40)

        for c in sorted(casc, key=lambda x: x["tag"]):
            if c.get("derived"):
                continue
            print(f"  info: {c['tag']:<30s} esc={c['cascade']['escalation_rate']:5.1f}% "
                  f"base F1={c['base_only']['f1']:6.2f} -> "
                  f"casc F1={c['cascade']['f1']:6.2f} "
                  f"({c['delta_f1_vs_base']:+.2f})  "
                  f"base@esc={c['base_on_escalated']['accuracy']:5.1f}% "
                  f"llm@esc={c['llm_on_escalated']['accuracy']:5.1f}%")


    # ------------------------------------------------------ faithfulness prose
    faith = load("faithfulness.json")
    if faith:
        by = {f["tag"]: f for f in faith}
        def fchk(tag, path, claimed, tol=0.005):
            if tag not in by:
                return
            node = by[tag]
            for k in path.split("."):
                node = node.get(k, {}) if isinstance(node, dict) else None
            check(f"{tag}.{path} ({claimed})", claimed, node, tol=tol)

        fchk("main_Test3_b20", "EGR.mean_egr", 0.974)
        fchk("main_Test3_b20", "EGR.hallucination_rate", 0.020)
        fchk("main_Test3_b20", "CFN.necessity_lift", 1.32, tol=0.01)
        fchk("main_Test3_b20", "CFN.mean_abs_shift", 0.222, tol=0.001)
        fchk("main_Test3_b20", "CFN.random_control_mean_abs_shift", 0.168, tol=0.001)
        fchk("main_Test3_b20", "CFN.flip_rate", 0.214, tol=0.005)
        fchk("main_Test3_b20", "CFN.random_control_flip_rate", 0.165, tol=0.005)
        fchk("main_Test3_b20", "AA.aa_precision_at_k", 0.366, tol=0.005)
        fchk("main_Test3_b20", "AA.random_baseline_precision", 0.275, tol=0.005)
        fchk("main_Test3_b20", "AA.aa_lift_over_chance", 1.33, tol=0.01)

        fchk("abl_nocontext_Test3", "EGR.mean_egr", 0.991)
        fchk("abl_nocontext_Test3", "EGR.fully_grounded_rate", 0.985, tol=0.005)
        fchk("abl_nocontext_Test3", "EGR.hallucination_rate", 0.008, tol=0.002)
        fchk("abl_nocontext_Test3", "CFN.necessity_lift", 1.62, tol=0.01)
        fchk("abl_nocontext_Test3", "AA.aa_lift_over_chance", 1.50, tol=0.01)

        fchk("abl_freeform_Test3", "EGR.mean_egr", 0.000, tol=0.001)
        fchk("abl_freeform_Test3", "EGR.hallucination_rate", 1.000, tol=0.001)
        if "abl_freeform_Test3" in by:
            check("freeform total citations (562)", 562,
                  by["abl_freeform_Test3"]["EGR"]["total_citations"], tol=0)

        fchk("abl_nocap_Test3", "EGR.mean_egr", 0.965)
        fchk("abl_nocap_Test3", "EGR.hallucination_rate", 0.026, tol=0.002)


        # Model-capacity sweep and the lenient-EGR refinement.
        fchk("sweep_phi4_14b_Test3", "EGR.mean_egr", 0.553)
        fchk("sweep_phi4_14b_Test3", "EGR.lenient_grounding_rate", 0.931)
        fchk("sweep_phi4_14b_Test3", "EGR.format_violation_rate", 0.388, tol=0.002)
        fchk("sweep_llama3.2_3b_Test3", "EGR.mean_egr", 0.939)
        fchk("abl_freeform_Test3", "EGR.lenient_grounding_rate", 0.044, tol=0.002)
        fchk("main_Test3_b20", "EGR.lenient_grounding_rate", 0.980)
        if "sweep_phi4_14b_Test3" in by:
            check("phi4 n completed (105)", 105, by["sweep_phi4_14b_Test3"]["n"], tol=0)

        # abstention endpoints quoted in the prose
        if "main_Test3_b20" in by:
            ab = {round(p["egr_threshold"], 2): p for p in by["main_Test3_b20"]["abstention"]}
            if 0.0 in ab:
                check("abstain err at tau=0 (37.5)", 37.5, ab[0.0]["error_on_kept"], tol=0.1)
            if 1.0 in ab:
                check("abstain err at tau=1 (36.7)", 36.7, ab[1.0]["error_on_kept"], tol=0.1)
                check("abstain coverage at tau=1 (94)", 94, ab[1.0]["coverage"], tol=0.6)


        # Validity check on EGR: are "hallucinations" merely artifacts withheld from
        # the bundle? Recomputed here so the claim in the prose cannot go stale.
        import common as _K
        _splits, _tot, _hidden = {}, 0, 0
        _allf = set(_K.BUNDLE_FEATURES)
        for _tag in ["main_Test3_b20", "main_Test1", "main_Test2",
                     "sweep_llama3.2_3b_Test3"]:
            _p = C.RESULTS / f"llm_{_tag}.json"
            if not _p.exists():
                continue
            _d = json.loads(_p.read_text())
            _s = _d["split"]
            _splits.setdefault(_s, _K.load(_s))
            for _r in _d["records"]:
                _row = _splits[_s].iloc[_r["row_index"]]
                for _c in _r["hallucinated"]:
                    _tot += 1
                    if _c in _allf and float(_row.get(_c, 0)) > 0:
                        _hidden += 1
        if _tot:
            check("EGR validity: present-but-unshown share (1.9%)", 1.9,
                  100 * _hidden / _tot, tol=0.15)
            check("EGR validity: true-invention share (98.1%)", 98.1,
                  100 * (_tot - _hidden) / _tot, tol=0.15)


    # ------------------------------------------------------ significance testing
    st = load("statistics.json")
    if st:
        for split, gain, lo, hi, pv in (("Test1", 6.51, 3.22, 9.87, 0.0001),
                                        ("Test2", 1.53, -0.68, 3.85, 0.2100),
                                        ("Test3", 2.30, 0.80, 3.90, 0.0167)):
            if split not in st:
                continue
            check(f"{split} gain ({gain})", gain, st[split]["gain"], tol=0.05)
            check(f"{split} CI low ({lo})", lo, st[split]["ci95"][0], tol=0.35)
            check(f"{split} CI high ({hi})", hi, st[split]["ci95"][1], tol=0.35)
            check(f"{split} McNemar p ({pv})", pv,
                  st[split]["mcnemar"]["p_value"], tol=0.02)
        # The central honesty claim: Test2 must remain non-significant.
        if "Test2" in st:
            if st["Test2"]["mcnemar"]["p_value"] < 0.05:
                FAILS.append("Test2 is now SIGNIFICANT - the report says it is not")
            if not (st["Test2"]["ci95"][0] < 0 < st["Test2"]["ci95"][1]):
                FAILS.append("Test2 CI no longer includes zero - report claim is stale")


    # -------------------------------------------------------- LLM-only ceiling
    ceil = load("ceiling.json")
    if ceil:
        for split, llm, base, delta in (("Test1", 58.67, 65.71, -7.05),
                                        ("Test2", 69.14, 72.50, -3.36),
                                        ("Test3", 36.36, 45.45, -9.09)):
            if split not in ceil:
                continue
            c = ceil[split]
            check(f"ceiling {split} LLM-only F1 ({llm})", llm,
                  c["llm_only"]["f1"], tol=0.05)
            check(f"ceiling {split} base F1 ({base})", base,
                  c["base_on_same_subsample"]["f1"], tol=0.05)
            check(f"ceiling {split} delta ({delta})", delta,
                  c["delta_llm_minus_base"], tol=0.05)
            check(f"ceiling {split} n sampled (400)", 400, c["n_sampled"], tol=0)
        # The claim is that escalating everything is NOT significantly better.
        for split in ("Test1", "Test2", "Test3"):
            if split in ceil and ceil[split]["mcnemar"]["p_value"] < 0.05:
                FAILS.append(f"ceiling {split} is now significant - report says it is not")
            if split in ceil and ceil[split]["delta_llm_minus_base"] > 0:
                FAILS.append(f"ceiling {split} now favours the LLM - report says otherwise")


    # ------------------------------------------------------ adversarial evaluation
    for split, ev5, fooled, egr_f, inj in (("Test1", 0.281, 0.618, 1.000, 19),
                                           ("Test3", 0.500, 0.974, 1.000, 2)):
        adv = load(f"adversarial_{split}.json")
        if not adv:
            continue
        check(f"adv {split} evasion @5 ({ev5})", ev5,
              adv["budgets"]["5"]["evasion_rate"], tol=0.002)
        L = adv.get("llm")
        if L:
            check(f"adv {split} LLM fooled rate ({fooled})", fooled,
                  L["llm_fooled_rate"], tol=0.002)
            check(f"adv {split} EGR when fooled ({egr_f})", egr_f,
                  L["mean_egr_when_fooled"], tol=0.002)
            check(f"adv {split} injected cited ({inj})", inj,
                  L["explanations_citing_injected"], tol=0)
            # The headline claim: grounding is NOT lower when the system is fooled.
            caught = L.get("mean_egr_when_caught")
            if caught is not None and L["mean_egr_when_fooled"] < caught - 0.001:
                FAILS.append(
                    f"adv {split}: EGR is now LOWER when fooled - the report claims otherwise")

    structural_checks(txt)
    report()


if __name__ == "__main__":
    main()
