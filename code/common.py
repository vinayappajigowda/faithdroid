"""Shared data loading and metric helpers.

Metrics are reported in exactly LAMD's format (F1 / FPR / FNR) so that the numbers in
this thesis sit beside the published ones without any recomputation.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config as C

# Every stage imports this module, so it is the one place to warn about a mismatched
# library stack before any number is computed.
try:
    import envcheck as _envcheck
    _envcheck.warn_if_mismatched()
except Exception:      # never let the guard itself break a run
    pass

META = json.loads((C.PROC / "meta.json").read_text())
FEATURE_GROUPS: dict[str, list[str]] = META["feature_columns"]
# Classifier feature space: named + unnamed syscalls, permissions, aggregates.
FEATURES: list[str] = (FEATURE_GROUPS["dynamic"] + FEATURE_GROUPS["static"]
                       + FEATURE_GROUPS["aggregate"])
# Artifact names that may appear in an LLM evidence bundle (human-meaningful only).
BUNDLE_FEATURES: list[str] = FEATURE_GROUPS["dynamic_named"] + FEATURE_GROUPS["static"]

SPLITS = ["train", "calib", "Test1", "Test2", "Test3"]
TEST_SPLITS = ["Test1", "Test2", "Test3"]


def load(split: str) -> pd.DataFrame:
    return pd.read_parquet(C.PROC / f"{split}.parquet")


def xy(df: pd.DataFrame, cols: list[str] | None = None):
    cols = cols or FEATURES
    return df[cols].to_numpy(dtype=np.float32), df["Malware"].to_numpy(dtype=int)


def metrics(y_true, y_pred, n_llm_calls: int = 0, n_total: int | None = None,
            abstained: int = 0) -> dict:
    """Confusion-derived metrics in LAMD's reporting format.

    FPR and FNR are reported because they, not accuracy, determine whether a triage
    system is deployable at scale.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    out = {
        "n": len(y_true),
        "accuracy": 100 * (tp + tn) / len(y_true) if len(y_true) else 0.0,
        "precision": 100 * prec,
        "recall": 100 * rec,
        "f1": 100 * f1,
        "fpr": 100 * fp / (fp + tn) if fp + tn else 0.0,
        "fnr": 100 * fn / (fn + tp) if fn + tp else 0.0,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }
    if n_total:
        out["llm_calls"] = n_llm_calls
        out["escalation_rate"] = 100 * n_llm_calls / n_total
        out["abstained"] = abstained
        out["coverage"] = 100 * (n_total - abstained) / n_total
    return out


def show(name: str, m: dict) -> None:
    line = (f"  {name:<26s} F1={m['f1']:6.2f}  P={m['precision']:6.2f}  "
            f"R={m['recall']:6.2f}  FPR={m['fpr']:5.2f}  FNR={m['fnr']:6.2f}  "
            f"acc={m['accuracy']:6.2f}")
    if "escalation_rate" in m:
        line += f"  esc={m['escalation_rate']:5.1f}%"
    print(line)


def save_json(obj, path) -> None:
    def _enc(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(str(type(o)))
    path.write_text(json.dumps(obj, indent=2, default=_enc))
    print(f"  -> {path}")
