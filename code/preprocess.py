"""Stage 1 - build the hybrid evidence base and prevalence-controlled temporal splits.

Why this file is not a two-line pandas concat:

1. KronoDroid's REAL device CSVs carry 484 columns mixing named system calls (dynamic
   artifacts), named permissions (static artifacts), aggregate counters, identifiers and
   VirusTotal-derived fields. The VT fields encode the label, so they are dropped.
2. `HighestModDate` needs cleaning: rows sit at the ZIP epoch (1980-01-01) or in the far
   future (2107) because APK internal modification dates are attacker-controllable.
3. Per-year class balance in KronoDroid swings from 88% benign (2011) to ~96% malware
   (2012). A naive temporal split would therefore measure *prior shift*, not concept
   drift. Every window is pinned to TARGET_PREVALENCE so the only thing that changes
   across windows is the concept.
4. Duplicate SHA-256 hashes must be removed or the same app can land in both train and
   test, silently inflating every number downstream.

Run:  python3 code/preprocess.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config as C


# --------------------------------------------------------------------------- load
def load_raw() -> pd.DataFrame:
    mal = pd.read_csv(C.MALWARE_CSV, low_memory=False)
    ben = pd.read_csv(C.BENIGN_CSV, low_memory=False)
    # `Malware` is already 1/0 in the respective files, but we set it explicitly
    # rather than trusting it.
    mal["Malware"] = 1
    ben["Malware"] = 0
    # Align on the intersection of columns so a schema mismatch surfaces loudly here
    # rather than as NaNs later.
    shared = [c for c in mal.columns if c in set(ben.columns)]
    dropped = (set(mal.columns) | set(ben.columns)) - set(shared)
    if dropped:
        print(f"  ! columns absent from one side, dropped: {sorted(dropped)}")
    df = pd.concat([mal[shared], ben[shared]], ignore_index=True)
    print(f"  loaded {len(df):,} rows x {df.shape[1]} cols "
          f"({int(df.Malware.sum()):,} malware / {int((df.Malware == 0).sum()):,} benign)")
    return df


def resolve_feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """Split columns into dynamic (system calls) and static (permissions) by position.

    KronoDroid lays the CSV out as:
        Package, Malware, <system calls...>, nr_syscalls, <permissions...>,
        nr_permissions, ..., metadata
    so the two aggregate counters act as reliable delimiters.
    """
    cols = list(df.columns)
    i_syscall_end = cols.index("nr_syscalls")
    i_perm_end = cols.index("nr_permissions")

    syscalls = [c for c in cols[:i_syscall_end] if c not in ("Package", "Malware")]
    perms = cols[i_syscall_end + 1:i_perm_end]

    # Guard against silent layout drift in a future dataset revision.
    assert syscalls, "no system-call columns resolved"
    assert perms, "no permission columns resolved"
    overlap = set(syscalls) & set(perms)
    assert not overlap, f"syscall/permission overlap: {overlap}"

    # Separate named syscalls from raw numeric ones. Both are used by the classifier;
    # only the named ones can appear in an LLM evidence bundle.
    named = [c for c in syscalls if not c.startswith(C.UNNAMED_SYSCALL_PREFIX)]
    unnamed = [c for c in syscalls if c.startswith(C.UNNAMED_SYSCALL_PREFIX)]

    agg = [c for c in C.AGG_COLS if c in df.columns]
    print(f"  dynamic (system calls): {len(syscalls)} "
          f"({len(named)} named, {len(unnamed)} unnamed SYS_nnn)")
    print(f"  static  (permissions) : {len(perms)}")
    print(f"  aggregate counters    : {len(agg)}")
    return {
        "dynamic": syscalls,
        "dynamic_named": named,
        "dynamic_unnamed": unnamed,
        "static": perms,
        "aggregate": agg,
    }


# ----------------------------------------------------------------------- cleaning
def clean(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)

    # Deduplicate on the app hash. Keeping duplicates would leak apps across splits.
    if "sha256" in df.columns:
        df = df.drop_duplicates(subset="sha256", keep="first")
        print(f"  deduplicated on sha256: {n0:,} -> {len(df):,} ({n0 - len(df):,} removed)")

    ts = pd.to_datetime(df[C.TIME_COL], errors="coerce", format="mixed")
    usable = ts.between(C.TIME_MIN, C.TIME_MAX)
    print(f"  {C.TIME_COL}: {usable.sum():,}/{len(df):,} rows inside "
          f"[{C.TIME_MIN}, {C.TIME_MAX}] ({100 * usable.mean():.1f}%) "
          f"- {(~usable).sum():,} dropped as epoch/corrupt")

    df = df.loc[usable].copy()
    df["ts"] = ts.loc[usable]
    df["year"] = df["ts"].dt.year
    return df


def to_numeric(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """Coerce every feature column to a finite float matrix."""
    sub = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    n_nan = int(sub.isna().sum().sum())
    if n_nan:
        print(f"  coerced {n_nan:,} non-numeric/missing feature cells to 0")
    return sub.fillna(0.0).astype(np.float32)


# ------------------------------------------------------------------------- splits
def pin_prevalence(df: pd.DataFrame, prevalence: float, rng: np.random.Generator,
                   label: str) -> pd.DataFrame:
    """Subsample so that malware makes up `prevalence` of the returned frame.

    Benign samples are the scarce class in most windows, so we keep all of them and
    subsample malware; if malware is scarce instead we do the reverse. Sampling is
    stratified by year so a window never collapses onto a single year.
    """
    mal = df[df.Malware == 1]
    ben = df[df.Malware == 0]
    if len(mal) == 0 or len(ben) == 0:
        raise ValueError(f"{label}: window has only one class "
                         f"(mal={len(mal)}, ben={len(ben)})")

    # n_mal / (n_mal + n_ben) = prevalence
    want_mal = int(round(len(ben) * prevalence / (1 - prevalence)))
    if want_mal <= len(mal):
        keep_mal, keep_ben = _stratified(mal, want_mal, rng), ben
    else:
        want_ben = int(round(len(mal) * (1 - prevalence) / prevalence))
        keep_mal, keep_ben = mal, _stratified(ben, want_ben, rng)

    out = pd.concat([keep_mal, keep_ben]).sample(frac=1.0, random_state=C.SEED)
    got = out.Malware.mean()
    print(f"  {label:6s} n={len(out):6,d}  mal={int(out.Malware.sum()):5,d}  "
          f"ben={int((out.Malware == 0).sum()):6,d}  prevalence={100 * got:.2f}%  "
          f"years={sorted(out.year.unique())}")
    return out


def _stratified(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Take n rows from df, spread proportionally across years."""
    n = min(n, len(df))
    if n == len(df):
        return df
    frac = n / len(df)
    parts = []
    for _, grp in df.groupby("year"):
        k = int(round(len(grp) * frac))
        if k > 0:
            parts.append(grp.sample(n=min(k, len(grp)), random_state=C.SEED))
    out = pd.concat(parts) if parts else df.sample(n=n, random_state=C.SEED)
    # Rounding can overshoot or undershoot the target; correct it.
    if len(out) > n:
        out = out.sample(n=n, random_state=C.SEED)
    elif len(out) < n:
        spare = df.drop(out.index)
        out = pd.concat([out, spare.sample(n=min(n - len(out), len(spare)),
                                          random_state=C.SEED)])
    return out


def build_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(C.SEED)
    splits: dict[str, pd.DataFrame] = {}

    print("\n  natural (uncontrolled) window sizes:")
    train_nat = df[df.year <= C.TRAIN_END]
    print(f"    train  <={C.TRAIN_END}: n={len(train_nat):,} "
          f"mal={int(train_nat.Malware.sum()):,} ben={int((train_nat.Malware == 0).sum()):,} "
          f"({100 * train_nat.Malware.mean():.1f}% mal)")
    for name, (lo, hi) in C.TEST_WINDOWS.items():
        w = df[(df.year >= lo) & (df.year <= hi)]
        print(f"    {name} {lo}-{hi}: n={len(w):,} mal={int(w.Malware.sum()):,} "
              f"ben={int((w.Malware == 0).sum()):,} ({100 * w.Malware.mean():.1f}% mal)")

    print(f"\n  prevalence-controlled at {100 * C.TARGET_PREVALENCE:.0f}% malware:")
    full_train = pin_prevalence(train_nat, C.TARGET_PREVALENCE, rng, "pool")

    # Stratified (not chronological) train/calibration split - see CALIB_FRACTION in
    # config.py for why a time-ordered sub-split is degenerate on this dataset.
    from sklearn.model_selection import train_test_split
    tr_idx, ca_idx = train_test_split(
        full_train.index,
        test_size=C.CALIB_FRACTION,
        stratify=full_train["Malware"],
        random_state=C.SEED,
    )
    splits["train"] = full_train.loc[tr_idx].copy()
    splits["calib"] = full_train.loc[ca_idx].copy()
    for nm in ("train", "calib"):
        s = splits[nm]
        print(f"  {nm:6s} n={len(s):6,d}  mal={int(s.Malware.sum()):5,d}  "
              f"prevalence={100 * s.Malware.mean():.2f}%  "
              f"ts=[{s.ts.min().date()} .. {s.ts.max().date()}]")

    for name, (lo, hi) in C.TEST_WINDOWS.items():
        w = df[(df.year >= lo) & (df.year <= hi)]
        splits[name] = pin_prevalence(w, C.TARGET_PREVALENCE, rng, name)

    return splits


# --------------------------------------------------------------------------- main
def main() -> None:
    print("=" * 74)
    print("FAITHDROID - Stage 1: hybrid evidence base + temporal splits")
    print("=" * 74)

    print("\n[1] loading KronoDroid REAL device set")
    df = load_raw()

    print("\n[2] resolving feature groups")
    groups = resolve_feature_groups(df)
    feat_cols = groups["dynamic"] + groups["static"] + groups["aggregate"]

    print("\n[3] cleaning")
    df = clean(df)

    print("\n[4] coercing features to numeric")
    X = to_numeric(df, feat_cols)
    df = pd.concat([df[[c for c in df.columns if c not in feat_cols]], X], axis=1)

    print("\n[5] building prevalence-controlled temporal splits")
    splits = build_splits(df)

    print("\n[6] writing parquet")
    keep_meta = ["Package", "sha256", "MalFamily", "Malware", "ts", "year"]
    for name, s in splits.items():
        cols = [c for c in keep_meta if c in s.columns] + feat_cols
        out = C.PROC / f"{name}.parquet"
        s[cols].to_parquet(out, index=False)
        print(f"  {out.name:14s} {len(s):6,d} rows  {out.stat().st_size / 1e6:6.1f} MB")

    meta = {
        "dataset": "KronoDroid REAL device (github.com/aleguma/kronodroid)",
        "time_column": C.TIME_COL,
        "time_bounds": [C.TIME_MIN, C.TIME_MAX],
        "target_prevalence": C.TARGET_PREVALENCE,
        "train_end_year": C.TRAIN_END,
        "test_windows": {k: list(v) for k, v in C.TEST_WINDOWS.items()},
        "calib_fraction": C.CALIB_FRACTION,
        "seed": C.SEED,
        "feature_groups": {k: len(v) for k, v in groups.items()},
        "feature_columns": groups,
        "split_sizes": {
            k: {"n": int(len(v)),
                "malware": int(v.Malware.sum()),
                "benign": int((v.Malware == 0).sum()),
                "prevalence": round(float(v.Malware.mean()), 4),
                "years": sorted(int(y) for y in v.year.unique())}
            for k, v in splits.items()
        },
    }
    (C.PROC / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  meta.json written ({len(feat_cols)} feature columns recorded)")
    print("\ndone.")


if __name__ == "__main__":
    main()
