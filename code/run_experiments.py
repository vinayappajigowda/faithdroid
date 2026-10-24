"""Drive every LLM configuration needed for the results section.

Responses are cached on disk, so re-invoking this script is cheap and it can be
interrupted and resumed without losing work.

Run:  python3 code/run_experiments.py            # main + ablations
      python3 code/run_experiments.py --ceiling  # add the 100%-escalation ceiling
"""
from __future__ import annotations

import argparse
import time

import common as K
import config as C
from llm_stage import run

# (tag, split, budget, model, mode, limit, cap_citations, sampling)
def plan(include_ceiling: bool) -> list[tuple]:
    jobs: list[tuple] = []

    # ---- headline: the cascade at a 10% escalation budget for Test1/Test2
    for split in ("Test1", "Test2"):
        jobs.append((f"main_{split}", split, 0.10, C.LLM_PRIMARY, "context", None,
                     True, "ranked"))

    # ---- Test3 is run once at the LARGEST budget. Escalation is a deterministic
    # ranking, so the top 10% is exactly a prefix of the top 20%: every smaller
    # budget on the Pareto curve is derived from this single run by cascade.py
    # rather than re-queried. This removes several hundred redundant LLM calls.
    jobs.append(("main_Test3_b20", "Test3", 0.20, C.LLM_PRIMARY, "context",
                 None, True, "ranked"))

    # ---- ablations on Test3 at a 10% budget, capped at 200 samples each.
    # EGR is a per-explanation mean, so n=200 gives a tight enough interval to
    # separate configurations whose grounding rates differ by more than a few points.
    ABL_N = 200
    # no retrieval context -> isolates the reference-distribution effect
    jobs.append(("abl_nocontext_Test3", "Test3", 0.10, C.LLM_PRIMARY,
                 "grounded", ABL_N, True, "ranked"))
    # no closed-vocabulary constraint -> expected to collapse EGR
    jobs.append(("abl_freeform_Test3", "Test3", 0.10, C.LLM_PRIMARY,
                 "freeform", ABL_N, True, "ranked"))
    # no citation length cap -> measures the prose-stuffing failure rate
    jobs.append(("abl_nocap_Test3", "Test3", 0.10, C.LLM_PRIMARY,
                 "context", ABL_N, False, "ranked"))

    # ---- model size sweep, capped to keep wall-clock sane
    for m, n in (("llama3.2:3b", 150), ("phi4:14b", 120)):
        jobs.append((f"sweep_{m.replace(':', '_')}_Test3", "Test3", 0.10, m,
                     "context", n, True, "ranked"))

    # ---- LLM-only ceiling: the LAMD/TraceRAG operating point (escalate everything).
    # Defined in run_ceiling.py so the two entry points cannot disagree about the
    # subsample size or the sampling scheme; they write the same files.
    if include_ceiling:
        from run_ceiling import ceiling_jobs
        jobs.extend(ceiling_jobs())

    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ceiling", action="store_true")
    a = ap.parse_args()

    jobs = plan(a.ceiling)
    print(f"{len(jobs)} configurations queued\n")
    t0 = time.time()
    for n, (tag, split, budget, model, mode, limit, cap, sampling) in enumerate(jobs, 1):
        out = C.RESULTS / f"llm_{tag}.json"
        print(f"\n{'=' * 74}\n[{n}/{len(jobs)}] {tag}\n{'=' * 74}")
        try:
            res = run(split, budget, model, mode, limit, cap_citations=cap,
                      sampling=sampling)
            K.save_json(res, out)
        except Exception as e:  # keep going; one bad config must not sink the sweep
            print(f"  !! {tag} failed: {type(e).__name__}: {e}")
        print(f"  cumulative wall clock: {(time.time() - t0) / 60:.1f} min")

    print(f"\nall done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
