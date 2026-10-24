"""Prevalence-aware retrieval of labelled neighbours for comparative LLM reasoning.

Motivation is empirical. With a bare evidence bundle, qwen2.5:7b returned MALWARE for
16 of 20 apps when only 1 was malicious, justifying itself with artifacts that are
near-universal in Android (INTERNET, ACCESS_NETWORK_STATE, WRITE_EXTERNAL_STORAGE, and
even VIBRATE and USE_FINGERPRINT). This is MalEval's "surface cues rather than verifiable
evidence" reproduced exactly: the model has no notion of what a *normal* app looks like,
so every permission reads as suspicious.

Supplying k nearest labelled neighbours from the training period fixes the missing
reference distribution. Retrieval is restricted to the training split, which ends in
2014 and is therefore strictly time-anterior to every test window - no label leakage.

Similarity is Jaccard over binarised artifact presence, which suits sparse binary
permission/syscall sets better than Euclidean distance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common as K


class NeighbourIndex:
    """Jaccard-nearest labelled neighbours drawn only from the training period.

    Jaccard similarity is computed as a single matrix-vector product rather than via
    sklearn's brute-force boolean metric, which evaluates in Python and took several
    seconds per query over 23,581 training rows - dominating total runtime. For binary
    vectors:

        intersection = M . v
        union        = |M| + |v| - intersection
        similarity   = intersection / union

    which is exact, not an approximation, and reduces the per-query cost to one BLAS
    call.
    """

    def __init__(self, k: int = 6):
        self.k = k
        self.cols = K.BUNDLE_FEATURES          # named syscalls + permissions
        self.train = K.load("train").reset_index(drop=True)
        self.M = (self.train[self.cols].to_numpy(dtype=np.float32) > 0)
        self.Mf = self.M.astype(np.float32)
        self.row_sums = self.Mf.sum(axis=1)
        self.labels = self.train["Malware"].to_numpy(dtype=int)
        self.families = self.train["MalFamily"].astype(str).to_numpy()
        # Precompute the present-artifact list per row once, so rendering is cheap.
        self._present: dict[int, list[str]] = {}

    def _artifacts(self, i: int) -> list[str]:
        if i not in self._present:
            self._present[i] = [c for c, on in zip(self.cols, self.M[i]) if on]
        return self._present[i]

    def query(self, row: pd.Series) -> list[dict]:
        v = (row[self.cols].to_numpy(dtype=np.float32) > 0).astype(np.float32)
        inter = self.Mf @ v
        union = self.row_sums + v.sum() - inter
        sim = np.divide(inter, union, out=np.zeros_like(inter),
                        where=union > 0)
        # Stable sort, not argpartition: similarity ties heavily here (52 of 58 sampled
        # queries picked a different equally-similar neighbour set under argpartition),
        # and neighbours go into the prompt, so unstable ties made the response cache key
        # depend on the numpy build. Costs ~7 ms per query instead of ~1 ms.
        # Changing this line invalidates every cached context-mode response.
        k = min(self.k, len(sim))
        cand = np.argsort(-sim, kind="stable")[:k]

        out = []
        for i in cand:
            present = self._artifacts(int(i))
            out.append({
                "similarity": float(sim[i]),
                "label": "MALWARE" if self.labels[i] == 1 else "BENIGN",
                "family": self.families[i] if self.labels[i] == 1 else "",
                "n_artifacts": len(present),
                "artifacts": present,
            })
        return out

    @staticmethod
    def render(neighbours: list[dict], max_artifacts: int = 12) -> str:
        """Format neighbours for the prompt, showing enough to compare against."""
        lines = []
        for n, nb in enumerate(neighbours, 1):
            fam = f", family={nb['family']}" if nb["family"] else ""
            shown = sorted(nb["artifacts"])[:max_artifacts]
            more = (f" (+{nb['n_artifacts'] - len(shown)} more)"
                    if nb["n_artifacts"] > len(shown) else "")
            lines.append(
                f"  [{n}] label={nb['label']}{fam}, "
                f"similarity={nb['similarity']:.2f}\n"
                f"      artifacts: {', '.join(shown)}{more}")
        return "\n".join(lines)

    @staticmethod
    def summarise(neighbours: list[dict]) -> str:
        n_mal = sum(nb["label"] == "MALWARE" for nb in neighbours)
        return (f"{n_mal} of {len(neighbours)} nearest known apps are MALWARE, "
                f"{len(neighbours) - n_mal} are BENIGN")
