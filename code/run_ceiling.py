"""LLM-only ceiling: the operating point of TraceRAG and LAMD, which escalate 100%.

Sampled uniformly at random rather than by escalation rank. The ranked selection is
deliberately biased toward uncertain samples, so measuring the ceiling on a ranked
prefix would understate what the reasoner achieves on the population as a whole.

400 applications per window keeps wall-clock tractable; the report states the subsample
size and treats the result as an estimate with a bootstrap interval rather than an exact
figure.
"""
import time
import common as K
import config as C
from llm_stage import run

N_PER_WINDOW = 400


def ceiling_jobs() -> list[tuple]:
    """The ceiling configuration, defined once and shared with run_experiments.py.

    Both entry points write llm_ceiling_<split>.json, so a second definition there (it
    used 350 samples and ranked sampling) meant whichever ran last decided the result.
    Layout: (tag, split, budget, model, mode, limit, cap_citations, sampling).
    """
    return [(f"ceiling_{s}", s, 1.00, C.LLM_PRIMARY, "context", N_PER_WINDOW,
             True, "random") for s in K.TEST_SPLITS]


if __name__ == "__main__":
    t0 = time.time()
    for split in K.TEST_SPLITS:
        print(f"\n{'='*70}\nceiling: {split}\n{'='*70}", flush=True)
        try:
            res = run(split, 1.00, C.LLM_PRIMARY, "context", N_PER_WINDOW,
                      cap_citations=True, sampling="random")
            K.save_json(res, C.RESULTS / f"llm_ceiling_{split}.json")
        except Exception as e:
            print(f"  !! {split} failed: {type(e).__name__}: {e}", flush=True)
        print(f"  cumulative: {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"\nceiling done in {(time.time()-t0)/60:.1f} min")
