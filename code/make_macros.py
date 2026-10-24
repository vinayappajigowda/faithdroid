"""Emit every prose number as a LaTeX macro, so the narrative cannot go stale.

Tables were already generated from the saved results; the numbers embedded in sentences
were not, and had to be retyped by hand whenever an experiment was rerun. That is the
only reason verify_report.py needed a hard-coded copy of each claim.

This writes report/text/generated_macros.tex, which the report inputs before any chapter.
After rerunning the pipeline, regenerate this file and the prose is correct by
construction.

Run:  python3 code/make_macros.py
"""
from __future__ import annotations

import json

import numpy as np

import common as K
import config as C

OUT = C.RESULTS.parent / "report" / "text" / "generated_macros.tex"
OUT_V2 = C.RESULTS.parent / "report_v2" / "text" / "generated_macros.tex"


def _load(name):
    p = C.RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def collect() -> dict[str, str]:
    """macro name -> already-formatted string. Names are letters only, as LaTeX requires."""
    m: dict[str, str] = {}

    def put(name: str, value, fmt: str = "{:.2f}") -> None:
        if value is None:
            return
        m[name] = fmt.format(value)

    # ------------------------------------------------------------- split geometry
    meta = json.loads((C.PROC / "meta.json").read_text())
    ss, fg = meta["split_sizes"], meta["feature_groups"]
    for key, macro in (("train", "NTrain"), ("calib", "NCalib"),
                       ("Test1", "NTestOne"), ("Test2", "NTestTwo"),
                       ("Test3", "NTestThree")):
        put(macro, ss[key]["n"], "{:,d}")
        put(macro + "Mal", ss[key]["malware"], "{:,d}")
    put("NDynamic", fg["dynamic"], "{:d}")
    put("NDynamicNamed", fg["dynamic_named"], "{:d}")
    put("NDynamicUnnamed", fg["dynamic_unnamed"], "{:d}")
    put("NStatic", fg["static"], "{:d}")
    put("NFeatures", fg["dynamic"] + fg["static"] + fg["aggregate"], "{:d}")

    # ------------------------------------------------------------------ baselines
    b = _load("baselines.json")
    if b:
        for model, tag in (("random_forest", "RF"), ("xgboost", "XGB"),
                           ("drebin_svm", "Drebin"), ("deepdrebin", "DeepDrebin")):
            for split, sp in (("Test1", "TestOne"), ("Test2", "TestTwo"),
                              ("Test3", "TestThree")):
                d = b.get(model, {}).get(split)
                if not d:
                    continue
                put(f"{tag}{sp}FOne", d["f1"])
                put(f"{tag}{sp}Recall", d["recall"])
                put(f"{tag}{sp}Precision", d["precision"])
                put(f"{tag}{sp}FPR", d["fpr"])

    # -------------------------------------------------------------------- routing
    r = _load("routing_diagnostic.json")
    if r:
        budgets = (("0.05", "BFive"), ("0.1", "BTen"),
                   ("0.2", "BTwenty"), ("0.3", "BThirty"))
        signals = (("margin_x_malware", "MarginXMal"), ("prob_margin", "ProbMargin"),
                   ("prob_malware", "ProbMal"), ("prob_entropy", "ProbEntropy"),
                   ("conformal_novelty", "ConfNovelty"),
                   ("conformal_combined", "ConfCombined"),
                   ("conformal_ratio", "ConfRatio"),
                   ("conformal_ambiguity", "ConfAmbiguity"))
        for sig, sname in signals:
            per_budget = []
            for bud, bname in budgets:
                if sig in r and bud in r[sig]:
                    v = float(np.mean(r[sig][bud]))
                    put(f"Lift{sname}{bname}", v)
                    per_budget.append(v)
            # The prose quotes each signal as a range "x--y depending on budget", so bind
            # the extremes rather than whichever budget happens to hold them today.
            if per_budget:
                put(f"Lift{sname}Min", min(per_budget))
                put(f"Lift{sname}Max", max(per_budget))
        # Prose quotes the conformal signals as a range rather than one by one.
        conf = [float(np.mean(r[s][b])) for s, _ in signals[4:]
                for b, _ in budgets if s in r and b in r[s]]
        if conf:
            put("ConfLiftMin", min(conf))
            put("ConfLiftMax", max(conf))

    # -------------------------------------------------------------------- cascade
    casc = _load("cascade.json")
    if casc:
        by = {c["tag"]: c for c in casc}
        for tag, name in (("main_Test1", "TestOne"), ("main_Test2", "TestTwo"),
                          ("main_Test3_b20", "TestThree")):
            c = by.get(tag)
            if not c:
                continue
            put(f"Casc{name}BaseFOne", c["base_only"]["f1"])
            put(f"Casc{name}FOne", c["cascade"]["f1"])
            put(f"Casc{name}Delta", c["delta_f1_vs_base"], "{:+.2f}")
            put(f"Casc{name}BandedDelta", c["delta_f1_banded_vs_base"], "{:+.2f}")
            put(f"Casc{name}BaseAtEsc", c["base_on_escalated"]["accuracy"], "{:.1f}")
            put(f"Casc{name}LLMAtEsc", c["llm_on_escalated"]["accuracy"], "{:.1f}")
            put(f"Casc{name}OverrideRate", c["banded_detail"]["override_rate"], "{:.1f}")
            put(f"Casc{name}BlindFNR", c["cascade"]["fnr"])
            put(f"Casc{name}BaseFNR", c["base_only"]["fnr"])
            put(f"Casc{name}BaseFPR", c["base_only"]["fpr"])
            put(f"Casc{name}NEscalated", c["n_escalated"], "{:d}")
            put(f"Casc{name}EscRate", c["cascade"]["escalation_rate"], "{:.1f}")

        # blind-override degradation curve on Test3
        for tag, name in (("main_Test3_b20@b2", "BTwo"), ("main_Test3_b20@b5", "BFive"),
                          ("main_Test3_b20@b10", "BTen"),
                          ("main_Test3_b20@b15", "BFifteen")):
            if tag in by:
                put(f"BlindTestThree{name}", by[tag]["delta_f1_vs_base"], "{:+.2f}")

        # Prose quotes the banded override rate as a range across the three windows.
        rates = [by[t]["banded_detail"]["override_rate"]
                 for t in ("main_Test1", "main_Test2", "main_Test3_b20") if t in by]
        if rates:
            put("OverrideRateMin", min(rates), "{:.1f}")
            put("OverrideRateMax", max(rates), "{:.1f}")

        ctx = [c for c in casc if c["mode"] == "context" and c["n_escalated"] > 50]
        if ctx:
            put("MeanPromptTokens",
                float(np.mean([c["cost"]["prompt_tokens"] / c["n_escalated"]
                               for c in ctx])), "{:.0f}")

    # --------------------------------------------------------------- faithfulness
    faith = _load("faithfulness.json")
    if faith:
        by = {f["tag"]: f for f in faith}
        cfgs = (("main_Test3_b20", "Main"), ("abl_nocontext_Test3", "NoContext"),
                ("abl_freeform_Test3", "Freeform"), ("abl_nocap_Test3", "NoCap"),
                ("sweep_phi4_14b_Test3", "PhiFour"),
                ("sweep_llama3.2_3b_Test3", "LlamaThreeB"),
                ("main_Test1", "MainTestOne"), ("main_Test2", "MainTestTwo"))
        for tag, name in cfgs:
            f = by.get(tag)
            if not f:
                continue
            E, F, A = f.get("EGR", {}), f.get("CFN", {}), f.get("AA", {})
            put(f"Egr{name}", E.get("mean_egr"), "{:.3f}")
            put(f"EgrLenient{name}", E.get("lenient_grounding_rate"), "{:.3f}")
            put(f"EgrFormatViol{name}", E.get("format_violation_rate"), "{:.3f}")
            put(f"Halluc{name}", E.get("hallucination_rate"), "{:.3f}")
            put(f"FullyGrounded{name}", E.get("fully_grounded_rate"), "{:.3f}")
            # Percent forms: the prose states several of these as percentages, and a
            # fraction-valued macro would not match the literal in the text.
            for src, mac in (("format_violation_rate", "EgrFormatViolPct"),
                             ("hallucination_rate", "HallucPct"),
                             ("fully_grounded_rate", "FullyGroundedPct")):
                if E.get(src) is not None:
                    put(f"{mac}{name}", 100 * E[src], "{:.1f}")
            for src, mac in (("flip_rate", "CfnFlipPct"),
                             ("random_control_flip_rate", "CfnRandFlipPct")):
                if F.get(src) is not None:
                    put(f"{mac}{name}", 100 * F[src], "{:.1f}")
            put(f"TotalCitations{name}", E.get("total_citations"), "{:d}")
            put(f"NecLift{name}", F.get("necessity_lift"))
            put(f"CfnShift{name}", F.get("mean_abs_shift"), "{:.3f}")
            put(f"CfnRandShift{name}", F.get("random_control_mean_abs_shift"), "{:.3f}")
            put(f"CfnFlip{name}", F.get("flip_rate"), "{:.3f}")
            put(f"CfnRandFlip{name}", F.get("random_control_flip_rate"), "{:.3f}")
            put(f"AaPrec{name}", A.get("aa_precision_at_k"), "{:.3f}")
            put(f"AaRand{name}", A.get("random_baseline_precision"), "{:.3f}")
            put(f"AaLift{name}", A.get("aa_lift_over_chance"))
            put(f"NSamples{name}", f.get("n"), "{:d}")
            # Verdict accuracy on the escalated subset, quoted in the model-size sweep.
            put(f"VerdictAcc{name}", f.get("verdict_metrics", {}).get("accuracy"),
                "{:.1f}")
            put(f"VerdictFOne{name}", f.get("verdict_metrics", {}).get("f1"))

        main = by.get("main_Test3_b20")
        if main:
            ab = {round(p["egr_threshold"], 2): p for p in main["abstention"]}
            if 0.0 in ab:
                put("AbstainErrTauZero", ab[0.0]["error_on_kept"], "{:.1f}")
            if 1.0 in ab:
                put("AbstainErrTauOne", ab[1.0]["error_on_kept"], "{:.1f}")
                put("AbstainCovTauOne", ab[1.0]["coverage"], "{:.0f}")

    # ------------------------------------------------------------- significance
    st = _load("statistics.json")
    if st:
        for split, name in (("Test1", "TestOne"), ("Test2", "TestTwo"),
                            ("Test3", "TestThree")):
            s = st.get(split)
            if not s:
                continue
            put(f"Gain{name}", s["gain"], "{:+.2f}")
            put(f"CILo{name}", s["ci95"][0], "{:+.2f}")
            put(f"CIHi{name}", s["ci95"][1], "{:+.2f}")
            put(f"PValue{name}", s["mcnemar"]["p_value"], "{:.4f}")

    # ------------------------------------------------------------------- ceiling
    ceil = _load("ceiling.json")
    if ceil:
        for split, name in (("Test1", "TestOne"), ("Test2", "TestTwo"),
                            ("Test3", "TestThree")):
            c = ceil.get(split)
            if not c:
                continue
            put(f"Ceil{name}LLMFOne", c["llm_only"]["f1"])
            put(f"Ceil{name}BaseFOne", c["base_on_same_subsample"]["f1"])
            put(f"Ceil{name}Delta", c["delta_llm_minus_base"], "{:+.2f}")
            put(f"Ceil{name}N", c["n_sampled"], "{:d}")

    # ------------------------------------------------- conformal validity / bands
    # These are written by conformal.py and cascade.py. Results produced before those
    # scripts persisted them simply yield no macros, which is why the corresponding
    # prose sites stay literal until the next full rerun.
    cg = _load("conformal_gate.json")
    if isinstance(cg, dict) and "validity" in cg:
        names = {"calib": "Calib", "Test1": "TestOne",
                 "Test2": "TestTwo", "Test3": "TestThree"}
        errs = []
        for split, sname in names.items():
            v = cg["validity"].get(split, {}).get("0.10")
            if not v:
                continue
            put(f"ConfErr{sname}", v["error_pct"])
            put(f"ConfEmpty{sname}", v["empty_pct"], "{:.1f}")
            if split != "calib":
                errs.append(v["error_pct"])
        if errs:
            put("ConfErrMin", min(errs))
            put("ConfErrMax", max(errs))

    ob = _load("override_bands.json")
    if ob:
        words = ["Zero", "One", "Two", "Three", "Four"]
        for i, band in enumerate(ob.get("per_band", [])):
            if i < len(words):
                put(f"BandAcc{words[i]}", band["accuracy"], "{:.3f}")
                put(f"BandN{words[i]}", band["n"], "{:,d}")

    # -------------------------------------------------------------- adversarial
    for split, name in (("Test1", "TestOne"), ("Test3", "TestThree")):
        adv = _load(f"adversarial_{split}.json")
        if not adv:
            continue
        # LaTeX control sequences may contain letters only - a digit terminates the name,
        # so \AdvEvasionTestOneK1 is not a definable command. Spell the budget out.
        for k, word in (("1", "One"), ("2", "Two"), ("5", "Five"),
                        ("10", "Ten"), ("20", "Twenty"), ("40", "Forty")):
            if k in adv.get("budgets", {}):
                put(f"AdvEvasion{name}K{word}",
                    100 * adv["budgets"][k]["evasion_rate"], "{:.1f}")
        L = adv.get("llm")
        if L:
            put(f"AdvFooled{name}", 100 * L["llm_fooled_rate"], "{:.1f}")
            put(f"AdvEgrFooled{name}", L["mean_egr_when_fooled"], "{:.3f}")
            put(f"AdvEgrCaught{name}", L.get("mean_egr_when_caught"), "{:.3f}")
            put(f"AdvInjectedCited{name}", L["explanations_citing_injected"], "{:d}")
            put(f"AdvNTested{name}", L["n_tested"], "{:d}")
            put(f"AdvFooledCount{name}", L["llm_fooled"], "{:d}")

    return m


def render(macros: dict[str, str]) -> str:
    lines = ["% Generated by code/make_macros.py - do not edit by hand.",
             "% Every number the report states in prose is defined here from the saved",
             "% results, so rerunning the pipeline updates the narrative automatically.",
             ""]
    for name in sorted(macros):
        lines.append(f"\\newcommand{{\\{name}}}{{{macros[name]}}}")
    return "\n".join(lines) + "\n"


def main() -> None:
    macros = collect()
    text = render(macros)
    OUT.write_text(text)
    print(f"wrote {OUT} ({len(macros)} macros)")
    if OUT_V2.parent.exists():
        OUT_V2.write_text(text)
        print(f"wrote {OUT_V2}")


if __name__ == "__main__":
    main()
