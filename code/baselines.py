"""Stage 2 - reproduce the learned baselines under temporal drift.

Four detectors, chosen to mirror the baselines LAMD reports so the comparison is direct:

  drebin_svm    linear SVM over binarised features   (Drebin, Arp et al.)
  deepdrebin    MLP over the same feature space      (DeepDrebin)
  random_forest RF over sensitive-artifact counts    (Malscan-style surrogate)
  xgboost       gradient boosting                    (strong modern tabular baseline)

Each is trained once on `train` and evaluated unchanged on Test1/2/3, so the numbers
show what drift does to a *static* model - which is the premise of the whole thesis.

Run:  python3 code/baselines.py
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

import common as K
import config as C

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def build_models() -> dict:
    """Each entry is (estimator, needs_binarised_input)."""
    return {
        # Drebin uses binary presence/absence of artifacts with a linear SVM.
        "drebin_svm": Pipeline([
            ("bin", Binarizer(threshold=0.0)),
            ("clf", LinearSVC(C=1.0, class_weight="balanced",
                              max_iter=5000, random_state=C.SEED)),
        ]),
        # DeepDrebin replaces the linear classifier with a small DNN on the same space.
        "deepdrebin": Pipeline([
            ("bin", Binarizer(threshold=0.0)),
            ("sc", StandardScaler(with_mean=False)),
            ("clf", MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=300,
                                  early_stopping=True, random_state=C.SEED)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample",
            n_jobs=-1, random_state=C.SEED),
        "xgboost": XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9,
            # 10% prevalence -> upweight the positive class by 9x
            scale_pos_weight=(1 - C.TARGET_PREVALENCE) / C.TARGET_PREVALENCE,
            eval_metric="logloss", n_jobs=-1, random_state=C.SEED,
            tree_method="hist"),
    }


def decision_scores(model, X) -> np.ndarray:
    """Uniform 'higher = more malicious' score, whatever the estimator exposes."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def main() -> None:
    print("=" * 78)
    print("FAITHDROID - Stage 2: learned baselines under temporal drift")
    print("=" * 78)

    tr = K.load("train")
    Xtr, ytr = K.xy(tr)
    print(f"\ntrain: n={len(ytr):,}  malware={ytr.sum():,} ({100 * ytr.mean():.2f}%)  "
          f"features={Xtr.shape[1]}")

    tests = {s: K.xy(K.load(s)) for s in K.TEST_SPLITS}
    results: dict[str, dict] = {}
    scores: dict[str, dict[str, list]] = {}

    for name, model in build_models().items():
        print(f"\n[{name}]")
        model.fit(Xtr, ytr)
        results[name] = {}
        scores[name] = {}
        for split, (Xte, yte) in tests.items():
            pred = model.predict(Xte)
            m = K.metrics(yte, pred)
            results[name][split] = m
            scores[name][split] = decision_scores(model, Xte).tolist()
            K.show(split, m)

    # ---------------------------------------------------------------- drift summary
    print("\n" + "=" * 78)
    print("F1 across drift windows (the premise: static models decay)")
    print("=" * 78)
    print(f"  {'model':<16s} {'Test1':>8s} {'Test2':>8s} {'Test3':>8s}   {'decay':>8s}")
    for name in results:
        f = [results[name][s]["f1"] for s in K.TEST_SPLITS]
        print(f"  {name:<16s} {f[0]:8.2f} {f[1]:8.2f} {f[2]:8.2f}   {f[0] - f[2]:+8.2f}")

    K.save_json(results, C.RESULTS / "baselines.json")
    K.save_json(scores, C.RESULTS / "baseline_scores.json")


if __name__ == "__main__":
    main()
