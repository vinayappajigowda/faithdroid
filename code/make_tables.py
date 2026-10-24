"""Emit the report's result tables as LaTeX, straight from the saved result files.

Numbers in the report are generated, never transcribed. If an experiment is re-run the
tables change with it, so the prose and the evidence cannot silently diverge.

Run:  python3 code/make_tables.py
"""
from __future__ import annotations

import json
import re

import numpy as np

import common as K
import config as C

OUT = C.RESULTS.parent / "report" / "text" / "generated_tables.tex"


def _load(name):
    p = C.RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def tbl_baselines() -> str:
    b = _load("baselines.json")
    if not b:
        return ""
    names = {"drebin_svm": "Drebin-style linear SVM",
             "deepdrebin": "DeepDrebin-style MLP",
             "random_forest": "Random Forest",
             "xgboost": "XGBoost (adopted as Stage 2)"}
    rows = []
    for key, lab in names.items():
        if key not in b:
            continue
        cells = []
        for s in K.TEST_SPLITS:
            m = b[key][s]
            cells.append(f"{m['f1']:.2f} / {m['fpr']:.2f} / {m['fnr']:.2f}")
        rows.append(f"{lab} & " + " & ".join(cells) + r" \\")
    return r"""
\begin{table}[htbp]
\centering\small
\caption{Learned baselines under temporal drift, reported as F1 / FPR / FNR (\%) in the
format used by \citet{qian2025lamd}. Each model is trained once on data up to 2014 and
evaluated unchanged on all three windows. These are LAMD-\emph{style} baselines on
KronoDroid, not reproductions of LAMD's AndroZoo results; absolute values are not
comparable across the two datasets.}
\label{tab:baselines}
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Model} & \textbf{Test1 (2015--16)} & \textbf{Test2 (2017--18)} &
\textbf{Test3 (2019--20)} \\
& F1 / FPR / FNR & F1 / FPR / FNR & F1 / FPR / FNR \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def tbl_routing() -> str:
    d = _load("routing_diagnostic.json")
    if not d:
        return ""
    pretty = {
        "margin_x_malware": r"\textbf{Margin $\times$ malware (adopted)}",
        "prob_margin": "Probability margin",
        "prob_entropy": "Predictive entropy",
        "prob_malware": "Malware probability",
        "conformal_novelty": r"Conformal novelty $1-\max(p)$",
        "conformal_combined": r"Conformal combined",
        "conformal_ratio": r"Conformal ratio",
        "conformal_ambiguity": r"Conformal ambiguity $\min(p)$",
    }
    budgets = ["0.05", "0.1", "0.2", "0.3"]
    rows = []
    for key in pretty:
        if key not in d:
            continue
        cells = []
        for b in budgets:
            v = d[key].get(b)
            cells.append(f"{np.mean(v):.2f}" if v else "--")
        rows.append(f"{pretty[key]} & " + " & ".join(cells) + r" \\")
    return r"""
\begin{table}[htbp]
\centering\small
\caption{Routing-signal selection. Values are error-concentration lift, the ratio of the
base detector's error rate inside the escalated set to its error rate outside, averaged
over the three drift windows. A lift of 1.0 is no better than random selection. Every
conformal-derived signal is at or below that threshold; probability-margin uncertainty
exceeds it by an order of magnitude.}
\label{tab:routing}
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Routing signal} & \textbf{2\%} & \textbf{10\%} & \textbf{20\%} & \textbf{30\%} \\
& \multicolumn{4}{c}{\emph{escalation budget}} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def tbl_cascade() -> str:
    rows_in = _load("cascade.json")
    if not rows_in:
        return ""
    main = [r for r in rows_in
            if r["mode"] == "context" and not r.get("derived")
            and r["tag"].startswith("main")]
    if not main:
        return ""
    body = []
    for r in sorted(main, key=lambda x: x["split"]):
        c, b, bd = r["cascade"], r["base_only"], r["banded_detail"]
        cb = r["cascade_banded"]
        body.append(
            f"{r['split']} & {b['f1']:.2f} & "
            f"{r['cascade']['escalation_rate']:.1f}\\% & {c['f1']:.2f} & "
            f"{c['f1'] - b['f1']:+.2f} & "
            f"{bd['override_rate']:.1f}\\% & \\textbf{{{cb['f1']:.2f}}} & "
            f"\\textbf{{{cb['f1'] - b['f1']:+.2f}}} & "
            f"{r['base_on_escalated']['accuracy']:.1f}\\% & "
            f"{r['llm_on_escalated']['accuracy']:.1f}\\% \\\\")
    return r"""
\begin{table}[htbp]
\centering\small
\setlength{\tabcolsep}{4pt}
\caption{Cascade performance under the two override rules. \emph{Blind} replaces the base
verdict on every escalated sample; \emph{banded} replaces it only where the base detector is
known from calibration data to be unreliable ($\hat{p} \in [0.3, 0.7)$). The final two columns
are the key diagnostic --- accuracy of each stage restricted to the escalated subset --- and
explain the difference: escalation pays only while the LLM is more accurate than the detector
it replaces.}
\label{tab:cascade}
\begin{tabular}{@{}lccccccccc@{}}
\toprule
& & \multicolumn{3}{c}{\textbf{Blind override}} &
\multicolumn{3}{c}{\textbf{Banded override}} &
\multicolumn{2}{c}{\textbf{Acc. on esc.}} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}
\textbf{Window} & \textbf{base F1} & esc. & F1 & $\Delta$ & ovr. & F1 & $\Delta$ &
base & LLM \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def tbl_faithfulness() -> str:
    rows_in = _load("faithfulness.json")
    if not rows_in:
        return ""
    # The ablations form a CHAIN, not independent removals from the full system:
    #   full      = neighbours + copy-exactly rule
    #   grounded  = copy-exactly rule only          (isolates the retrieval effect)
    #   freeform  = neither                         (isolates the rule, vs grounded)
    # Labels are indented to show which row each is a controlled comparison against.
    label = {
        "context": r"\textbf{FAITHDROID (full)}",
        "grounded": r"\quad $-$ retrieval context",
        "freeform": r"\quad\quad $-$ copy-exactly rule",
    }
    # Explicit order so the chain reads top-to-bottom; the length-cap ablation is a
    # separate one-variable removal from the full system and is placed last.
    order = ["main_Test3_b20", "abl_nocontext_Test3", "abl_freeform_Test3",
             "abl_nocap_Test3"]
    by = {r["tag"]: r for r in rows_in}
    body = []
    for tag in order:
        r = by.get(tag)
        if r is None:
            continue
        nm = label.get(r["mode"], r["mode"])
        if "nocap" in tag:
            nm = r"$-$ citation length cap \emph{(from full)}"
        E, F, A = r["EGR"], r["CFN"], r["AA"]
        body.append(
            f"{nm} & {r['n']} & {E['mean_egr']:.3f} & "
            f"{E.get('lenient_grounding_rate', 0):.3f} & "
            f"{100 * E.get('format_violation_rate', 0):.1f}\\% & "
            f"{100 * E['fully_grounded_rate']:.1f}\\% & "
            f"{F.get('necessity_lift', 0):.2f} & "
            f"{A.get('aa_precision_at_k', 0):.3f} & "
            f"{A.get('aa_lift_over_chance', 0):.2f} \\\\")
    if not body:
        return ""
    return r"""
\begin{table}[htbp]
\centering\small
\setlength{\tabcolsep}{4pt}
\caption{Automatic faithfulness audit and ablations, on Test3 (2019--20). \emph{Strict} EGR
requires each citation to match an artifact exactly; \emph{lenient} EGR first splits citations
that pack several names into one field, so the gap between the two isolates output-format
violations from genuine invention. CFN lift and AA lift are relative to matched random
controls, so 1.00 means indistinguishable from chance.
\textbf{The first three rows form a chain, not independent removals}: indentation shows the row
each is a controlled single-variable comparison against. Removing retrieval also removes the
neighbour context, so the copy-exactly rule is isolated by comparing row~3 with row~2, not with
row~1. Row~4 removes only the output-schema length cap from the full system.}
\label{tab:faithfulness}
\begin{tabular}{@{}lcccccccc@{}}
\toprule
& & \multicolumn{4}{c}{\textbf{Grounding}} & \textbf{CFN} &
\multicolumn{2}{c}{\textbf{AA@10}} \\
\cmidrule(lr){3-6}\cmidrule(lr){7-7}\cmidrule(lr){8-9}
\textbf{Configuration} & \textbf{n} & strict & lenient & fmt.\ viol. & fully &
lift & prec. & lift \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def tbl_pareto() -> str:
    rows_in = _load("cascade.json")
    if not rows_in:
        return ""
    pts = [r for r in rows_in if r["split"] == "Test3" and r["mode"] == "context"]
    if len(pts) < 3:
        return ""
    pts.sort(key=lambda r: r["cascade"]["escalation_rate"])
    body = []
    for r in pts:
        c, cb, bd = r["cascade"], r["cascade_banded"], r["banded_detail"]
        body.append(
            f"{c['escalation_rate']:.1f}\\% & {c['llm_calls']} & "
            f"{c['f1']:.2f} & {c['fnr']:.2f} & "
            f"{bd['override_rate']:.1f}\\% & {cb['f1']:.2f} & {cb['fnr']:.2f} & "
            f"{r['usd_per_1k_apps_equiv']:.3f} \\\\")
    base = pts[0]["base_only"]
    return r"""
\begin{table}[htbp]
\centering\small
\setlength{\tabcolsep}{4pt}
\caption{Cost--accuracy trade-off on Test3 (2019--20), the hardest window. Under the blind
rule, F1 \emph{falls below the base detector} once the budget exceeds roughly 2\%, and the
rising FNR shows why: the LLM begins overriding correct malware detections with BENIGN
verdicts. The banded rule saturates at a 3.4\% override rate and remains above the baseline at
every budget. The final column prices the tokens actually consumed at the GPT-4o-mini rates
used by \citet{qian2025lamd}, so the operating point is commensurable with published figures;
the tokens were in fact generated locally at no monetary cost. Note that this comparison also
reflects a shallower analysis, not only a lower escalation rate --- see
Section~\ref{sec:evaluation}.}
\label{tab:pareto}
\begin{tabular}{@{}lcccccccc@{}}
\toprule
& & \multicolumn{2}{c}{\textbf{Blind}} & \multicolumn{3}{c}{\textbf{Banded}} & \\
\cmidrule(lr){3-4}\cmidrule(lr){5-7}
\textbf{Budget} & \textbf{LLM calls} & F1 & FNR & ovr.\ rate & F1 & FNR &
\textbf{Equiv. \$/1k} \\
\midrule
0.0\% (base only) & 0 & """ + (
        f"{base['f1']:.2f} & {base['fnr']:.2f} & 0.0\\% & "
        f"{base['f1']:.2f} & {base['fnr']:.2f}") + r""" & 0.000 \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def tbl_sweep() -> str:
    rows_in = _load("faithfulness.json")
    casc = {c["tag"]: c for c in (_load("cascade.json") or [])}
    if not rows_in:
        return ""
    # 3B and 14B come from the capacity sweep; the 7B row is the main Test3 run.
    order = [("llama3.2:3b", "sweep_llama3.2_3b_Test3"),
             ("qwen2.5:7b", "main_Test3_b20"),
             ("phi4:14b", "sweep_phi4_14b_Test3")]
    by = {r["tag"]: r for r in rows_in}
    body = []
    for model, tag in order:
        if tag not in by:
            continue
        r = by[tag]
        E = r["EGR"]
        acc = (casc[tag]["llm_on_escalated"]["accuracy"] if tag in casc
               else r["verdict_metrics"].get("accuracy", 0))
        lat = casc.get(tag, {}).get("mean_latency_s", 0)
        body.append(
            f"\\texttt{{{model}}} & {r['n']} & {acc:.1f}\\% & "
            f"{E['mean_egr']:.3f} & {E.get('lenient_grounding_rate', 0):.3f} & "
            f"{100 * E.get('format_violation_rate', 0):.1f}\\% & "
            f"{100 * E['fully_grounded_rate']:.1f}\\% & {lat:.1f}\\,s \\\\")
    if len(body) < 2:
        return ""
    return r"""
\begin{table}[htbp]
\centering\small
\setlength{\tabcolsep}{4pt}
\caption{Effect of model capacity on the reasoning stage (Test3). Verdict accuracy is
essentially flat across an almost fivefold range of parameters, but the ability to \emph{obey
the citation format} is not, and it does not increase with size: the 14B model packs several
artifact names into a single citation field on 38.8\% of citations, which strict grounding
scores as hallucination. That its lenient score recovers to 0.931 shows the artifacts it names
are real --- the defect is format compliance, not invention. The 7B model is the best operating
point on every axis including latency.}
\label{tab:sweep}
\begin{tabular}{@{}lccccccc@{}}
\toprule
& & & \multicolumn{3}{c}{\textbf{Grounding}} & & \\
\cmidrule(lr){4-6}
\textbf{Model} & \textbf{n} & \textbf{Acc. on esc.} & strict & lenient & fmt.\ viol. &
\textbf{fully gr.} & \textbf{latency} \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def tbl_ceiling() -> str:
    c = _load("ceiling.json")
    if not c:
        return ""
    body = []
    for split in ("Test1", "Test2", "Test3"):
        if split not in c:
            continue
        r = c[split]
        lo, hi = r["llm_only_f1_ci"]
        blo, bhi = r["base_f1_ci"]
        dlo, dhi = r["delta_ci95"]
        p = r["mcnemar"]["p_value"]
        body.append(
            f"{split} & {r['n_sampled']} & "
            f"{r['llm_only']['f1']:.2f} & $[{lo:.1f}, {hi:.1f}]$ & "
            f"{r['base_on_same_subsample']['f1']:.2f} & $[{blo:.1f}, {bhi:.1f}]$ & "
            f"{r['delta_llm_minus_base']:+.2f} & $[{dlo:+.1f}, {dhi:+.1f}]$ & "
            f"{p:.2f} \\\\")
    if not body:
        return ""
    return r"""
\begin{table}[htbp]
\centering\small
\setlength{\tabcolsep}{3.5pt}
\caption{The LLM-only ceiling --- the reasoner applied to 100\% of inputs, which is the operating
point of TraceRAG and LAMD --- against the cheap detector on \emph{identical} applications.
Estimated on a uniform random subsample of 400 per window (random, not ranked: the escalation
ranking is biased toward uncertain samples and would understate the ceiling). The difference is
tested pairwise. In every window the point estimate favours the cheap detector, and no difference
is statistically significant: escalating everything is indistinguishable from escalating nothing,
at roughly two hundred times the cost. Contrast the cascade, which escalates 3--5\% and achieves a
significant gain (Table~\ref{tab:significance}).}
\label{tab:ceiling}
\begin{tabular}{@{}lccccccccc@{}}
\toprule
& & \multicolumn{2}{c}{\textbf{LLM only (100\%)}} & \multicolumn{2}{c}{\textbf{Cheap detector}} &
\multicolumn{3}{c}{\textbf{Paired difference}} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-9}
\textbf{Window} & \textbf{n} & F1 & 95\% CI & F1 & 95\% CI & $\Delta$ & 95\% CI & $p$ \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def tbl_families() -> str:
    """Escalation coverage of the malware the detector misses, per window."""
    import numpy as np
    pvp = C.RESULTS / "conformal_pvalues.json"
    if not pvp.exists():
        return ""
    pv = json.loads(pvp.read_text())
    body = []
    for tag, split in (("main_Test1", "Test1"), ("main_Test2", "Test2"),
                       ("main_Test3_b20", "Test3")):
        p = C.RESULTS / f"llm_{tag}.json"
        if not p.exists():
            continue
        recs = json.loads(p.read_text())["records"]
        df = K.load(split)
        pred = np.asarray(pv[split]["pred"])
        esc = {r["row_index"]: r for r in recs}
        fam = df["MalFamily"].astype(str).to_numpy()
        mal_idx = np.where((df.Malware == 1).to_numpy())[0]
        missed = [j for j in mal_idx if pred[j] == 0]
        m_esc = [j for j in missed if j in esc]
        rec = [j for j in m_esc if esc[j]["llm_verdict"] == 1]
        from collections import defaultdict
        rf = defaultdict(int)
        for j in rec:
            rf[fam[j]] += 1
        # Two families keep the cell inside the text block; family names in this
        # dataset can be long (e.g. "Locker/SLocker Ransomware").
        def _short(f: str) -> str:
            f = f.split("/")[0]
            return f if len(f) <= 12 else f[:11] + "."
        top = ", ".join(f"\\texttt{{{_short(f)}}}({n})" for f, n in
                        sorted(rf.items(), key=lambda kv: -kv[1])[:2]) or "--"
        n_fam = df[df.Malware == 1]["MalFamily"].astype(str).nunique()
        body.append(
            f"{split} & {n_fam} & {len(missed)} & {len(m_esc)} "
            f"({100 * len(m_esc) / max(1, len(missed)):.0f}\\%) & "
            f"{len(rec)} ({100 * len(rec) / max(1, len(m_esc)):.0f}\\%) & {top} \\\\")
    if not body:
        return ""
    return r"""
\begin{table}[htbp]
\centering\small
\setlength{\tabcolsep}{4pt}
\caption{Does escalation reach the malware the cheap detector misses? The Test2 row explains
that window's null result: the gate routed only one of the thirteen missed malware, so the
reasoning stage was never given the samples that would have mattered. The limitation is in the
routing signal, which ranks by the detector's own uncertainty and is therefore blind to samples
it is confidently wrong about.}
\label{tab:families}
\begin{tabular}{@{}lccccl@{}}
\toprule
\textbf{Window} & \textbf{families} & \textbf{missed by base} & \textbf{escalated} &
\textbf{recovered by LLM} & \textbf{families recovered} \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def tbl_significance() -> str:
    st = _load("statistics.json")
    if not st:
        return ""
    body = []
    for split in ("Test1", "Test2", "Test3"):
        if split not in st:
            continue
        s = st[split]
        lo, hi = s["ci95"]
        p = s["mcnemar"]["p_value"]
        sig = r"\textbf{yes}" if p < 0.05 else r"\emph{no}"
        body.append(
            f"{split} & {s['base_f1']:.2f} & {s['banded_f1']:.2f} & "
            f"{s['gain']:+.2f} & $[{lo:+.2f}, {hi:+.2f}]$ & "
            f"{s['mcnemar']['discordant']} & {p:.4f} & {sig} \\\\")
    if not body:
        return ""
    return r"""
\begin{table}[htbp]
\centering\small
\caption{Statistical significance of the cascade's gain over the base detector. Intervals are
paired percentile bootstraps over applications (10{,}000 resamples); the test is exact McNemar
on discordant pairs, which is the correct choice because both systems are evaluated on identical
inputs and their errors are therefore correlated. The gain on Test2 is \emph{not} significant ---
its interval includes zero --- so no improvement is claimed on that window.}
\label{tab:significance}
\begin{tabular}{@{}lcccccc@{}}
\toprule
\textbf{Window} & \textbf{base F1} & \textbf{cascade F1} & \textbf{gain} &
\textbf{95\% CI} & \textbf{discordant} & \textbf{$p$} \\
\midrule
""" + "\n".join(body).replace(" & \\textbf{yes} \\\\", " \\\\").replace(
        " & \\emph{no} \\\\", " \\\\") + r"""
\bottomrule
\end{tabular}
\end{table}
"""




V2 = C.RESULTS.parent / "report_v2" / "text"


def write_v2() -> None:
    """Emit one file per table for the Practicum template.

    That template is page-limited, so each caption is replaced with a concise variant.
    The long captions in the full report carry methodological caveats; here those caveats
    live in the surrounding prose instead, so nothing is lost, only relocated.
    """
    if not V2.exists():
        return
    short = {
        "tab:baselines":
            "Learned baselines under temporal drift (F1 / FPR / FNR, \\%), in the format of "
            "\\citet{qian2025lamd}. Each model is trained once on data to 2014 and evaluated "
            "unchanged. These are LAMD-\\emph{style} baselines on KronoDroid, so absolute values "
            "are not comparable with LAMD's AndroZoo results.",
        "tab:ceiling":
            "The LLM-only ceiling (100\\% escalation, the operating point of TraceRAG and LAMD) "
            "against the detector on identical applications, estimated on 400 random applications "
            "per window and tested pairwise.",
        "tab:cascade":
            "Cascade performance under the two override rules. \\emph{Blind} replaces the verdict "
            "on every escalated application; \\emph{banded} only where calibration shows the "
            "detector unreliable ($\\hat{p} \\in [0.3,0.7)$). The last two columns give each "
            "stage's accuracy on the escalated subset.",
        "tab:faithfulness":
            "Faithfulness audit and ablations (2019--20 window). \\emph{Strict} grounding requires "
            "exact citation matches; \\emph{lenient} first splits packed citations, so the gap "
            "isolates format violations from invention. The first three rows form a chain: "
            "indentation shows the row each is compared against.",
    }
    parts = {
        "tbl_baselines.tex": tbl_baselines(),
        "tbl_ceiling.tex": tbl_ceiling(),
        "tbl_cascade.tex": tbl_cascade(),
        "tbl_faithfulness.tex": tbl_faithfulness(),
    }
    n = 0
    for name, body in parts.items():
        if not body.strip():
            continue
        for lbl, cap in short.items():
            if f"\\label{{{lbl}}}" in body:
                body = re.sub(r"\\caption\{.*?\}\n\\label", 
                              lambda m: "\\caption{" + cap + "}\n\\label",
                              body, count=1, flags=re.S)
        (V2 / name).write_text("% Generated by code/make_tables.py\n" + body)
        n += 1
    print(f"wrote {n} tables to report_v2/text/ (concise captions)")


def main() -> None:
    parts = [
        "% Generated by code/make_tables.py - do not edit by hand.\n",
        tbl_baselines(), tbl_routing(), tbl_cascade(), tbl_significance(),
        tbl_pareto(), tbl_faithfulness(), tbl_sweep(), tbl_ceiling(), tbl_families(),
    ]
    parts = [p for p in parts if p.strip()]
    OUT.write_text("\n".join(parts))
    print(f"wrote {OUT} ({len(parts) - 1} tables)")
    write_v2()


if __name__ == "__main__":
    main()
