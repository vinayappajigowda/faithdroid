"""Generate every figure for the report from the saved result files.

Each figure degrades gracefully if its inputs are missing, so this can be run while
the experiment sweep is still in flight.

Run:  python3 code/figures.py
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import common as K
import config as C

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})
INK, ACC, WARN, OK = "#1f2933", "#2b6cb0", "#c05621", "#2f855a"


def _load(name):
    p = C.RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def _save(fig, name):
    out = C.IMAGES / name
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> images/{name}")


# ------------------------------------------------------------------ fig 1: system
def fig_architecture():
    W, H = 13.0, 6.4
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, title, body, fc, ec, tsize=9.5, bsize=7.3):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                    fc=fc, ec=ec, lw=1.5))
        ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top",
                fontsize=tsize, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h / 2 - 0.16, body, ha="center", va="center",
                fontsize=bsize, color=INK, linespacing=1.5)

    def arrow(x1, y1, x2, y2, label="", color=INK, dy=0.16):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=13, lw=1.5, color=color))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, label, ha="center",
                    fontsize=7.4, color=color, fontweight="bold")

    ROW_Y, ROW_H = 3.15, 2.75

    box(0.15, ROW_Y, 2.25, ROW_H, "S1  Evidence base",
        "KronoDroid REAL\n78,137 apps\n\n218 named syscalls\n(dynamic)\n"
        "166 permissions\n(static)\n+70 SYS_nnn, 21 agg.\n\n"
        "timestamped →\ndrift splits",
        "#f7fafc", "#cbd5e0")

    box(2.75, ROW_Y, 2.35, ROW_H, "S2  Cheap detector",
        "XGBoost\n475 features\n\ntrain ≤ 2014\nn=23,581\n10% malware\n\n"
        "+ Mondrian\nconformal calibration\nn=10,107",
        "#ebf8ff", ACC)

    box(5.5, 4.55, 2.15, 1.35, "Gate",
        "uncertainty score\n(1−2|p̂−0.5|)(0.5+p̂)\n\ntop-k by budget",
        "#fffaf0", WARN)

    box(5.5, 3.15, 2.15, 1.15, "Retained  ~90%",
        "base verdict kept\nzero LLM cost",
        "#f0fff4", OK)

    box(8.05, ROW_Y, 2.35, ROW_H, "S3  LLM reasoner",
        "local Ollama\nqwen2.5:7b — $0\n\nCLOSED evidence\nbundle\n\n"
        "+ k=6 labelled\nneighbours\n(≤2014, no leakage)\n\n"
        "JSON-schema\nconstrained output",
        "#faf5ff", "#6b46c1")

    box(10.8, ROW_Y, 2.05, ROW_H, "S4  Audit",
        "EGR\nevidence grounding\n\nCFN\ncounterfactual\nnecessity\n\n"
        "AA\nagreement vs SHAP\n\n→ abstain if\nEGR < τ",
        "#fff5f7", "#b83280")

    box(4.4, 1.05, 6.0, 1.15, "Final decision",
        "LLM verdict on escalated  ∪  base verdict on retained\n"
        "abstain where grounding fails",
        "#f7fafc", "#cbd5e0")

    arrow(2.40, 4.5, 2.75, 4.5)
    arrow(5.10, 4.6, 5.50, 5.05, "escalate", WARN)
    arrow(5.10, 4.4, 5.50, 3.80, "retain", OK, dy=-0.30)
    arrow(7.67, 5.05, 8.05, 4.6, "~10%", WARN)
    arrow(10.40, 4.5, 10.80, 4.5)
    arrow(9.22, ROW_Y, 9.22, 2.20)
    arrow(6.57, 3.15, 6.57, 2.20)

    ax.text(W / 2, 0.62, "Grounding is checkable by construction: the citation "
                         "vocabulary is closed, so hallucination is measurable "
                         "without human labels.",
            ha="center", fontsize=8.5, style="italic", color=INK)
    ax.text(W / 2, 0.24, "FAITHDROID", ha="center", fontsize=12,
            fontweight="bold", color=INK)
    _save(fig, "fig1_architecture.png")


# ------------------------------------------------------------------ fig 2: drift
def fig_baseline_drift():
    b = _load("baselines.json")
    if not b:
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = np.arange(len(K.TEST_SPLITS))
    labels = {"drebin_svm": "Drebin-style SVM", "deepdrebin": "DeepDrebin-style MLP",
              "random_forest": "Random Forest", "xgboost": "XGBoost"}
    for name, lab in labels.items():
        if name not in b:
            continue
        f1 = [b[name][s]["f1"] for s in K.TEST_SPLITS]
        ax.plot(x, f1, marker="o", lw=1.8, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(["Test1\n2015–16", "Test2\n2017–18", "Test3\n2019–20"])
    ax.set_ylabel("F1 (%)")
    ax.set_title("Learned detectors across temporal drift windows\n"
                 "(all windows pinned to 10% malware prevalence)", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_ylim(0, 100)
    ax.annotate("RF collapse\nrecall 9.8%", xy=(2, 17.3), xytext=(1.35, 33),
                fontsize=7.5, color=WARN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=WARN, lw=1.2))
    _save(fig, "fig2_baseline_drift.png")


# -------------------------------------------------------- fig 3: routing negative
def fig_routing():
    d = _load("routing_diagnostic.json")
    if not d:
        return
    budgets = ["0.05", "0.1", "0.2", "0.3"]
    budgets = [b for b in budgets if any(b in v for v in d.values())]
    order = sorted(d.items(), key=lambda kv: -np.mean(list(kv[1].values())[1]))
    names = [n for n, _ in order]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    x = np.arange(len(names))
    w = 0.8 / max(1, len(budgets))
    for i, b in enumerate(budgets):
        vals = [np.mean(d[n][b]) if b in d[n] else 0 for n in names]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w,
               label=f"budget {float(b) * 100:.0f}%")
    ax.axhline(1.0, color=WARN, ls="--", lw=1.4)
    ax.text(len(names) - 0.5, 1.35, "lift = 1: no better than random",
            fontsize=7.5, color=WARN, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7)
    ax.set_ylabel("error-concentration lift")
    ax.set_title("Negative result: every conformal signal fails as a routing score.\n"
                 "Margin-based uncertainty is stable at 13–14× across all budgets",
                 fontsize=9.5)
    ax.legend(fontsize=7.5)
    _save(fig, "fig3_routing_signals.png")


# -------------------------------------------------------------- fig 4: conformal
def fig_conformal_validity():
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    # numbers from conformal.py's validity table
    splits = ["calib", "Test1", "Test2", "Test3"]
    e05 = [4.97, 26.06, 26.21, 29.18]
    e10 = [9.97, 39.58, 40.23, 46.88]
    x = np.arange(len(splits))
    ax.bar(x - 0.2, e05, 0.4, label="ε = 0.05", color=ACC)
    ax.bar(x + 0.2, e10, 0.4, label="ε = 0.10", color="#90cdf4")
    ax.axhline(5, color=ACC, ls=":", lw=1.2)
    ax.axhline(10, color="#90cdf4", ls=":", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(splits)
    ax.set_ylabel("empirical error of prediction set (%)")
    ax.set_title("Conformal validity holds in-period, breaks under drift\n"
                 "(dotted lines = nominal significance)", fontsize=9.5)
    ax.legend(fontsize=8)
    _save(fig, "fig4_conformal_validity.png")


# ----------------------------------------------------------------- fig 5: pareto
def fig_pareto():
    rows = _load("cascade.json")
    if not rows:
        return
    pts = [r for r in rows if r["mode"] == "context" and r["split"] == "Test3"]
    if len(pts) < 2:
        return
    pts.sort(key=lambda r: r["cascade"]["escalation_rate"])
    esc = [r["cascade"]["escalation_rate"] for r in pts]
    f1_blind = [r["cascade"]["f1"] for r in pts]
    f1_band = [r["cascade_banded"]["f1"] for r in pts]
    base = pts[0]["base_only"]["f1"]

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(esc, f1_blind, marker="o", lw=1.9, color=WARN,
            label="blind override (replace on every escalated sample)")
    ax.plot(esc, f1_band, marker="s", lw=1.9, color=ACC,
            label="banded override (replace only where base is unreliable)")
    ax.axhline(base, color=INK, ls="--", lw=1.4,
               label=f"base detector alone (F1={base:.1f})")
    ax.fill_between(esc, base, f1_blind,
                    where=[v < base for v in f1_blind],
                    color=WARN, alpha=0.12)
    ax.set_xlabel("LLM escalation budget (% of inputs)")
    ax.set_ylabel("F1 (%)")
    ax.set_title("Cost–accuracy trade-off, Test3 (2019–20)\n"
                 "blind override falls below the baseline; banded override does not",
                 fontsize=9.5)
    ax.legend(fontsize=7.5, loc="lower left")
    ax.annotate("blind override\nharms here", xy=(esc[-1], f1_blind[-1]),
                xytext=(esc[-1] - 6, f1_blind[-1] + 6), fontsize=7.5,
                color=WARN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=WARN, lw=1.2))
    _save(fig, "fig5_pareto.png")


# ----------------------------------------------------------- fig 6: faithfulness
def fig_faithfulness():
    rows = _load("faithfulness.json")
    if not rows:
        return
    # Only the Test3 ablation chain, at full sample size. The n=20 smoke-test runs are
    # excluded: plotting them beside n=200+ results invites reading noise as signal.
    pretty = {
        "main_Test3_b20":      "FAITHDROID (full)\nn=570",
        "abl_nocontext_Test3": "$-$ retrieval context\nn=200",
        "abl_freeform_Test3":  "$-$ copy-exactly rule\nn=199",
        "abl_nocap_Test3":     "$-$ citation length cap\nn=200",
    }
    order = ["main_Test3_b20", "abl_nocap_Test3", "abl_nocontext_Test3",
             "abl_freeform_Test3"]
    by = {r["tag"]: r for r in rows if r.get("EGR")}
    rows = [by[t] for t in order if t in by]
    if len(rows) < 2:
        return
    rows = rows[::-1]                      # barh draws bottom-up
    names = [pretty[r["tag"]] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))

    axes[0].barh(names, [r["EGR"]["mean_egr"] for r in rows], color=ACC)
    axes[0].set_xlabel("mean EGR"); axes[0].set_xlim(0, 1.05)
    axes[0].set_title("Evidence Grounding Rate\n(hallucination, lower = worse)",
                      fontsize=9)

    axes[1].barh(names, [r["CFN"].get("necessity_lift", 0) for r in rows],
                 color="#6b46c1")
    axes[1].axvline(1.0, color=WARN, ls="--", lw=1.3)
    axes[1].set_xlabel("necessity lift vs random ablation")
    axes[1].set_title("Counterfactual Necessity\n(>1 = cited evidence matters more)",
                      fontsize=9)

    axes[2].barh(names, [r["AA"].get("aa_lift_over_chance", 0) for r in rows],
                 color="#b83280")
    axes[2].axvline(1.0, color=WARN, ls="--", lw=1.3)
    axes[2].set_xlabel("AA lift over chance")
    axes[2].set_title("Attribution Agreement vs SHAP\n(the XAI↔LLM bridge)",
                      fontsize=9)
    for a in axes:
        a.tick_params(labelsize=7)
    # The zero bars for the no-rule ablation are the headline: without the constraint
    # there is no citation to measure, so all three metrics are undefined-as-zero.
    for a, lab in zip(axes, ("EGR = 0.000", "undefined", "undefined")):
        a.annotate(lab, xy=(0, 0), xytext=(6, -2), textcoords="offset points",
                   fontsize=7, color=WARN, fontweight="bold", va="center")
    fig.tight_layout()
    _save(fig, "fig6_faithfulness.png")


# ------------------------------------------------------------- fig 7: abstention
def fig_abstention():
    rows = _load("faithfulness.json")
    if not rows:
        return
    cand = [r for r in rows if r.get("abstention") and r["mode"] == "context"]
    if not cand:
        return
    r = max(cand, key=lambda x: x["n"])
    cur = [p for p in r["abstention"] if p.get("n_kept")]
    if len(cur) < 2:
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    ax.plot([p["coverage"] for p in cur], [p["error_on_kept"] for p in cur],
            marker="o", lw=1.9, color="#b83280")
    for p in cur:
        ax.annotate(f"τ={p['egr_threshold']:.2f}",
                    (p["coverage"], p["error_on_kept"]),
                    textcoords="offset points", xytext=(6, 5), fontsize=7)
    ax.set_xlabel("coverage (% of escalated samples answered)")
    ax.set_ylabel("error rate on answered (%)")
    ax.set_title(f"Coverage–risk under EGR-gated abstention\n({r['tag']}, n={r['n']})",
                 fontsize=9.5)
    _save(fig, "fig7_abstention.png")



# ------------------------------------------------------- fig 8: adversarial
def fig_adversarial():
    import glob
    files = sorted(glob.glob(str(C.RESULTS / "adversarial_*.json")))
    if not files:
        return
    data = {}
    for f in files:
        d = json.loads(open(f).read())
        data[d["split"]] = d
    if not data:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    # left: evasion rate vs attack budget
    ax = axes[0]
    for split, d in sorted(data.items()):
        ks = sorted(int(k) for k in d["budgets"])
        rates = [100 * d["budgets"][str(k)]["evasion_rate"] for k in ks]
        ax.plot(ks, rates, marker="o", lw=1.9, label=split)
    ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10, 20, 40])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("benign-looking artifacts added by the attacker")
    ax.set_ylabel("detector evasion rate (%)")
    ax.set_title("Feature-addition attack evades the detector\n"
                 "(additions only \u2014 no malicious capability removed)", fontsize=9.5)
    ax.legend(fontsize=8)

    # right: grounding when fooled vs when caught
    ax = axes[1]
    labels, fooled, caught, nf, nc = [], [], [], [], []
    for split, d in sorted(data.items()):
        L = d.get("llm")
        if not L or L.get("mean_egr_when_fooled") is None:
            continue
        labels.append(f"{split}\n({L['n_tested']} evaded)")
        fooled.append(L["mean_egr_when_fooled"])
        caught.append(L.get("mean_egr_when_caught") or 0)
        nf.append(L["llm_fooled"]); nc.append(L["llm_caught"])
    if labels:
        x = np.arange(len(labels))
        ax.bar(x - 0.2, fooled, 0.4, color=WARN, label="both stages fooled")
        ax.bar(x + 0.2, caught, 0.4, color=OK, label="LLM caught the evasion")
        # Per-bar n, because Test3's "caught" bar rests on a single sample and must
        # not be read as comparable to the others.
        for xi, (v, n) in enumerate(zip(fooled, nf)):
            ax.text(xi - 0.2, v + 0.03, f"n={n}", ha="center", fontsize=7.5)
        for xi, (v, n) in enumerate(zip(caught, nc)):
            ax.text(xi + 0.2, v + 0.03, f"n={n}", ha="center", fontsize=7.5,
                    fontweight="bold" if n < 5 else "normal",
                    color=WARN if n < 5 else INK)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("mean Evidence Grounding Rate")
        ax.set_ylim(0, 1.15)
        ax.axhline(1.0, color=INK, ls=":", lw=1)
        ax.set_title("Grounding is HIGHEST when the verdict is WRONG\n"
                     "faithfulness does not imply correctness", fontsize=9.5)
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    _save(fig, "fig8_adversarial.png")

def main() -> None:
    print("generating figures")
    for f in (fig_architecture, fig_baseline_drift, fig_routing,
              fig_conformal_validity, fig_pareto, fig_faithfulness,
              fig_abstention, fig_adversarial):
        try:
            f()
        except Exception as e:
            print(f"  ! {f.__name__}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
