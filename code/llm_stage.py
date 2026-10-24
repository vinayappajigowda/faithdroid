"""Stage 4 - LLM reasoning over a closed evidence bundle.

The design decision that makes this thesis's central metric possible:

    The model is shown a CLOSED set of artifact identifiers drawn from the app itself,
    and is required to cite evidence using only strings from that set.

Because the permitted vocabulary is finite and known, every citation can be checked
against the app's actual feature vector *without any human ground truth*. That is what
makes Evidence Grounding Rate computable at scale, and it is precisely what TraceRAG
(which needed Google Android Security engineers on 24 apps) and RAG-Android (where a
hallucinated description is indistinguishable from a true one) cannot do.

Only human-meaningful artifacts enter the bundle: named system calls and named
permissions. The 70 unnamed `SYS_nnn` columns are used by the classifier but excluded
here, since "SYS_342" is not something a model or an analyst can reason about.

Responses are cached on disk keyed by (model, prompt-mode, sha256) so reruns are free.

Run:  python3 code/llm_stage.py --split Test3 --budget 0.10
"""
from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

import common as K
import config as C

# ------------------------------------------------------------------ output schema
# Enforced by Ollama's structured-output mode, so the parse cannot drift.
def verdict_schema(max_citation_len: int | None = 64) -> dict:
    """Structured-output schema.

    `max_citation_len` caps each citation's length. Without it, models stuff
    explanatory prose into `cited_evidence` ("As this permission is often used by
    malware to intercept user data...") - a hallucination mode EGR detected
    automatically on the first run. The uncapped variant is retained as an ablation so
    the rate of that behaviour can be reported.
    """
    item: dict = {"type": "string"}
    if max_citation_len:
        item["maxLength"] = max_citation_len
    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["MALWARE", "BENIGN"]},
            "cited_evidence": {
                "type": "array", "items": item, "minItems": 1, "maxItems": 6,
            },
            "reasoning": {"type": "string"},
        },
        "required": ["verdict", "cited_evidence", "reasoning"],
    }

GROUNDED_INSTRUCTION = """You are an Android malware analyst.

Decide whether the application described below is MALWARE or BENIGN.

EVIDENCE BUNDLE - this is the complete list of observed artifacts for this app:
{bundle}

RULES:
- Cite between 1 and 6 items in "cited_evidence".
- Every cited item MUST be copied EXACTLY, character for character, from the
  PERMISSIONS_REQUESTED or SYSCALLS_OBSERVED lists above.
- Do NOT put explanations in "cited_evidence" - put those in "reasoning".
- Base your verdict only on the artifacts shown."""

# The prompt that supplies a reference distribution. Added after measuring that a bare
# bundle produced 16/20 MALWARE verdicts on a set containing 1 malware sample.
CONTEXT_INSTRUCTION = """You are an Android malware analyst.

Decide whether the TARGET application is MALWARE or BENIGN.

IMPORTANT BASE RATE: in this population only about {prevalence:.0f}% of applications are
malware. The large majority are benign. Common permissions such as INTERNET,
ACCESS_NETWORK_STATE, WRITE_EXTERNAL_STORAGE, READ_PHONE_STATE and VIBRATE appear in
most benign applications and are NOT on their own evidence of malice. Judge the target
by how it compares with the known apps below, not by whether its permissions sound
alarming in isolation.

KNOWN APPS MOST SIMILAR TO THE TARGET (ground-truth labels, all from before 2015):
{neighbours}

NEIGHBOUR SUMMARY: {neighbour_summary}

TARGET APPLICATION - complete list of observed artifacts:
{bundle}

RULES:
- Cite between 1 and 6 items in "cited_evidence".
- Every cited item MUST be copied EXACTLY, character for character, from the TARGET's
  PERMISSIONS_REQUESTED or SYSCALLS_OBSERVED lists.
- Do NOT put explanations in "cited_evidence" - put those in "reasoning".
- Prefer artifacts that DISTINGUISH the target from the benign neighbours."""

# Ablation prompt: no closed-vocabulary constraint. Used to demonstrate that the
# grounding constraint - not the model - is what produces faithful citations.
FREEFORM_INSTRUCTION = """You are an Android malware analyst.

Decide whether the application described below is MALWARE or BENIGN.

OBSERVED ARTIFACTS:
{bundle}

Cite between 1 and 6 pieces of evidence supporting your verdict in "cited_evidence",
and explain your reasoning."""

MODES = ("grounded", "context", "freeform")


def build_bundle(row: pd.Series, top_perms: int = C.BUNDLE_TOP_PERMS,
                 top_syscalls: int = C.BUNDLE_TOP_SYSCALLS) -> tuple[str, set[str]]:
    """Render one app as a closed evidence bundle.

    Returns the prompt text and the set of legal citation strings, which is the
    ground truth against which EGR is later computed.
    """
    perms = [p for p in K.FEATURE_GROUPS["static"] if float(row.get(p, 0)) > 0]
    sysc = {s: float(row.get(s, 0)) for s in K.FEATURE_GROUPS["dynamic_named"]
            if float(row.get(s, 0)) > 0}

    # Permissions are binary presence; keep a stable alphabetical prefix.
    perms_shown = sorted(perms)[:top_perms]
    # System calls are invocation counts; the most-invoked are the most behaviourally
    # informative, so rank by count.
    sysc_shown = [s for s, _ in sorted(sysc.items(), key=lambda kv: -kv[1])[:top_syscalls]]

    allowed = set(perms_shown) | set(sysc_shown)
    bundle = (
        f"PERMISSIONS_REQUESTED ({len(perms_shown)} of {len(perms)} shown):\n"
        f"  {', '.join(perms_shown) if perms_shown else '(none)'}\n\n"
        f"SYSCALLS_OBSERVED ({len(sysc_shown)} of {len(sysc)} shown, "
        f"most-invoked first):\n"
        f"  {', '.join(sysc_shown) if sysc_shown else '(none)'}\n\n"
        f"SUMMARY COUNTERS:\n"
        f"  total_permissions={int(row.get('total_perm', 0))}, "
        f"dangerous_permissions={int(row.get('dangerous', 0))}, "
        f"distinct_syscalls={int(row.get('nr_syscalls', 0))}, "
        f"intent_filters={int(row.get('TotalIntentFilters', 0))}, "
        f"services={int(row.get('NrServices', 0))}"
    )
    return bundle, allowed


# ------------------------------------------------------------------------- ollama
def _cache_path(model: str, mode: str, key: str, cap: bool = True):
    safe = model.replace(":", "_").replace("/", "_")
    # The cap flag MUST be part of the key. It changes the output schema but not the
    # prompt, so without it the uncapped ablation would silently replay the capped
    # run's cached responses and measure nothing at all.
    suffix = "" if cap else "__nocap"
    return C.CACHE / f"{safe}__{mode}{suffix}__{key}.json"


def query(prompt: str, model: str, mode: str, cache_key: str,
          retries: int = 3, cap_citations: bool = True) -> dict | None:
    cp = _cache_path(model, mode, cache_key, cap_citations)
    if cp.exists():
        try:
            d = json.loads(cp.read_text())
            d["cached"] = True
            return d
        except json.JSONDecodeError:
            cp.unlink(missing_ok=True)  # corrupt cache entry, refetch

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": verdict_schema(64 if cap_citations else None),
        "options": {"temperature": C.LLM_TEMPERATURE,
                    "num_predict": C.LLM_NUM_PREDICT},
    }
    req = urllib.request.Request(
        C.OLLAMA_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})

    for attempt in range(retries):
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=C.LLM_TIMEOUT) as r:
                raw = json.loads(r.read())
            parsed = json.loads(raw["response"])
            out = {
                "verdict": parsed["verdict"],
                "cited_evidence": parsed.get("cited_evidence", []),
                "reasoning": parsed.get("reasoning", ""),
                "latency_s": round(time.time() - t0, 2),
                "eval_count": raw.get("eval_count", 0),
                "prompt_eval_count": raw.get("prompt_eval_count", 0),
                "cached": False,
            }
            cp.write_text(json.dumps(out))
            return out
        except (json.JSONDecodeError, KeyError, urllib.error.URLError,
                TimeoutError, OSError) as e:
            if attempt == retries - 1:
                print(f"    ! failed after {retries} attempts: {type(e).__name__}: {e}")
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


# --------------------------------------------------------------------------- main
def select_escalated(df: pd.DataFrame, budget: float,
                     sampling: str = "ranked") -> np.ndarray:
    """Indices to send to the LLM.

    `sampling="ranked"` takes the top-k by the routing signal validated in F3 - the
    cascade's actual behaviour.

    `sampling="random"` draws uniformly at random instead. This is required for the
    LLM-only ceiling: that reference point asks how the reasoner performs on the
    population as a whole, and the ranked selection is deliberately biased toward
    uncertain samples, so measuring the ceiling on a ranked prefix would understate it.
    """
    pv = json.loads((C.RESULTS / "conformal_pvalues.json").read_text())
    split = df.attrs["split"]
    proba = np.asarray(pv[split]["proba"])
    assert len(proba) == len(df), (
        f"{split}: {len(proba)} stored scores vs {len(df)} rows - rerun conformal.py")
    k = max(1, int(round(budget * len(df))))
    if sampling == "random":
        rng = np.random.default_rng(C.SEED)
        return np.sort(rng.choice(len(df), size=min(k, len(df)), replace=False))
    score = (1.0 - np.abs(proba - 0.5) * 2.0) * (0.5 + proba)
    return np.argsort(-score, kind="stable")[:k]


def run(split: str, budget: float, model: str, mode: str, limit: int | None,
        cap_citations: bool = True, sampling: str = "ranked") -> dict:
    df = K.load(split)
    df.attrs["split"] = split
    idx = select_escalated(df, budget, sampling=sampling)
    if limit:
        idx = idx[:limit]

    instruction = {"grounded": GROUNDED_INSTRUCTION,
                   "context": CONTEXT_INSTRUCTION,
                   "freeform": FREEFORM_INSTRUCTION}[mode]

    index = None
    if mode == "context":
        from retrieval import NeighbourIndex
        print("  building neighbour index over the training period...")
        index = NeighbourIndex(k=6)

    print(f"\n[{model} | {mode}] {split}: escalating {len(idx)} of {len(df)} "
          f"({100 * len(idx) / len(df):.1f}%)")

    # Build every prompt first, then issue requests through a small thread pool.
    # Ollama serves one request per slot, so concurrency mainly overlaps prompt
    # evaluation with generation; measured at roughly a 20% throughput gain.
    jobs = []
    for rank, i in enumerate(idx):
        row = df.iloc[i]
        bundle, allowed = build_bundle(row)
        if mode == "context":
            nbs = index.query(row)
            prompt = instruction.format(
                bundle=bundle,
                prevalence=100 * C.TARGET_PREVALENCE,
                neighbours=index.render(nbs),
                neighbour_summary=index.summarise(nbs))
        else:
            prompt = instruction.format(bundle=bundle)
        key = hashlib.sha256(
            (str(row.get("sha256", i)) + prompt).encode()).hexdigest()[:24]
        jobs.append((rank, int(i), row, allowed, prompt, key))

    t_start = time.time()
    done = 0
    results: dict[int, dict] = {}

    def _work(job):
        rank, i, row, allowed, prompt, key = job
        return rank, job, query(prompt, model, mode, key, cap_citations=cap_citations)

    with ThreadPoolExecutor(max_workers=C.LLM_WORKERS) as pool:
        for rank, job, res in pool.map(_work, jobs):
            done += 1
            if res is None:
                continue
            _, i, row, allowed, _, _ = job
            cited = res["cited_evidence"]
            grounded = [c for c in cited if c in allowed]
            results[rank] = {
                "row_index": i,
                "sha256": str(row.get("sha256", "")),
                "family": str(row.get("MalFamily", "")),
                "y_true": int(row["Malware"]),
                "llm_verdict": 1 if res["verdict"] == "MALWARE" else 0,
                "cited": cited,
                "grounded": grounded,
                "hallucinated": [c for c in cited if c not in allowed],
                "n_allowed": len(allowed),
                "egr": len(grounded) / len(cited) if cited else 0.0,
                "reasoning": res["reasoning"],
                "latency_s": res.get("latency_s", 0.0),
                "eval_count": res.get("eval_count", 0),
                "prompt_eval_count": res.get("prompt_eval_count", 0),
                "cached": bool(res.get("cached")),
            }
            if done % 50 == 0 or done == len(jobs):
                el = time.time() - t_start
                egrs = [r["egr"] for r in results.values()]
                print(f"    {done}/{len(jobs)}  elapsed={el:6.1f}s  "
                      f"cached={sum(r['cached'] for r in results.values())}  "
                      f"mean_EGR={np.mean(egrs):.3f}", flush=True)

    # Restore escalation rank order, which the Pareto prefix derivation depends on.
    records = [results[r] for r in sorted(results)]
    n_cached = sum(r["cached"] for r in records)

    out = {
        "split": split, "model": model, "mode": mode, "budget": budget,
        "sampling": sampling,
        "n_escalated": len(records), "n_total": len(df),
        "wall_clock_s": round(time.time() - t_start, 1),
        "n_cached": n_cached,
        "records": records,
    }
    if records:
        y = np.array([r["y_true"] for r in records])
        p = np.array([r["llm_verdict"] for r in records])
        out["llm_only_on_escalated"] = K.metrics(y, p)
        out["mean_egr"] = float(np.mean([r["egr"] for r in records]))
        out["fully_grounded_rate"] = float(np.mean(
            [r["egr"] == 1.0 for r in records]))
        fresh = [r for r in records if r["latency_s"] > 0]
        out["mean_latency_s"] = float(np.mean([r["latency_s"] for r in fresh])) if fresh else 0.0
        print(f"  verdict on escalated subset: F1={out['llm_only_on_escalated']['f1']:.2f}  "
              f"acc={out['llm_only_on_escalated']['accuracy']:.2f}")
        print(f"  mean EGR={out['mean_egr']:.3f}  "
              f"fully-grounded={100 * out['fully_grounded_rate']:.1f}%  "
              f"mean latency={out['mean_latency_s']:.2f}s")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="Test3")
    ap.add_argument("--budget", type=float, default=0.10)
    ap.add_argument("--model", default=C.LLM_PRIMARY)
    ap.add_argument("--mode", default="context", choices=list(MODES))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--sampling", default="ranked", choices=["ranked", "random"],
                    help="'random' is required for the LLM-only ceiling")
    ap.add_argument("--no-cap", action="store_true",
                    help="remove the citation length cap (prose-stuffing ablation)")
    a = ap.parse_args()

    res = run(a.split, a.budget, a.model, a.mode, a.limit,
              cap_citations=not a.no_cap, sampling=a.sampling)
    tag = a.tag or f"{a.split}_{a.model.replace(':', '_')}_{a.mode}_b{int(100 * a.budget)}"
    K.save_json(res, C.RESULTS / f"llm_{tag}.json")


if __name__ == "__main__":
    main()
