"""Environment guard - detect the library drift that silently changes every result.

The pipeline is deterministic given a fixed library stack, and not otherwise. Splits
rebuild identically anywhere, but the fitted models do not: on scikit-learn 1.9 rather
than the pinned 1.5.1, XGBoost F1 moved +2.50 / -1.67 / +3.56 across the three windows.
That detector chooses which samples escalate, so the cached LLM responses stop matching
too (hit rate fell to 25-44%), and the fitted override band moved.

The invariant tests all still pass on a drifted stack, correctly - no leakage is
introduced. Only verify_report.py catches it, at the very end, and blames the report
prose rather than the environment. This states the real cause early.

Run:  python3 code/envcheck.py          (exit 1 on mismatch)
      python3 code/envcheck.py --warn   (never exits non-zero)
"""
from __future__ import annotations

import argparse
import re
import sys
from importlib.metadata import PackageNotFoundError, version

import config as C

REQUIREMENTS = C.ROOT / "requirements.txt"
REFERENCE_PYTHON = (3, 13)


def parse_requirements(path=REQUIREMENTS) -> dict[str, str]:
    """Read the `name==version` pins, ignoring comments and blank lines."""
    pins: dict[str, str] = {}
    if not path.exists():
        return pins
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        m = re.fullmatch(r"([A-Za-z0-9_.\-]+)\s*==\s*([0-9A-Za-z_.\-]+)", line)
        if m:
            pins[m.group(1).lower().replace("_", "-")] = m.group(2)
    return pins


def check(verbose: bool = True) -> list[str]:
    """Return a list of human-readable mismatches; empty means the stack matches."""
    problems: list[str] = []
    pins = parse_requirements()

    if verbose:
        print("=" * 78)
        print("FAITHDROID - environment check")
        print("=" * 78)

    py = sys.version_info[:2]
    py_ok = py == REFERENCE_PYTHON
    if verbose:
        print(f"\n  {'python':<16s} {'.'.join(map(str, py)):<12s} "
              f"expected {'.'.join(map(str, REFERENCE_PYTHON)):<12s} "
              f"{'ok' if py_ok else 'MISMATCH'}")
    if not py_ok:
        problems.append(
            f"python {'.'.join(map(str, py))} != reference "
            f"{'.'.join(map(str, REFERENCE_PYTHON))}")

    if not pins:
        problems.append(f"no version pins found in {REQUIREMENTS}")
        if verbose:
            print(f"\n  ! could not read pins from {REQUIREMENTS}")
        return problems

    if verbose:
        print()
    for pkg, want in sorted(pins.items()):
        try:
            got = version(pkg)
        except PackageNotFoundError:
            got = None
        ok = got == want
        if verbose:
            print(f"  {pkg:<16s} {str(got or 'ABSENT'):<12s} expected {want:<12s} "
                  f"{'ok' if ok else 'MISMATCH'}")
        if not ok:
            problems.append(f"{pkg} {got or 'absent'} != pinned {want}")

    return problems


def report(problems: list[str], strict: bool) -> int:
    if not problems:
        print("\nenvironment matches the pinned stack - results are reproducible\n")
        return 0

    print(f"\n{len(problems)} ENVIRONMENT MISMATCH(ES):")
    for p in problems:
        print(f"  - {p}")
    print(
        "\nThe pipeline will still run and every invariant test will still pass, but the\n"
        "fitted models - and therefore every reported number - will differ from the\n"
        "published results. Cached LLM responses will also stop matching, because the\n"
        "escalated set is chosen from the detector's probabilities.\n"
        "\nTo reproduce the reported results, install the pinned stack:\n"
        f"    pip install -r {REQUIREMENTS}\n"
        "on Python "
        f"{'.'.join(map(str, REFERENCE_PYTHON))}.\n"
        "\nTo generate NEW results on this stack instead, rerun the pipeline from\n"
        "preprocess.py through to verify_report.py, and regenerate the LLM cache with a\n"
        "running Ollama server. Do not mix: results/ must come from one stack only.\n")
    return 1 if strict else 0


def warn_if_mismatched() -> None:
    """Cheap banner for import-time use by pipeline stages."""
    problems = check(verbose=False)
    if problems:
        print("  ! environment does not match requirements.txt "
              f"({len(problems)} mismatch(es)) - results will not reproduce the "
              "published numbers. Run `python3 code/envcheck.py` for detail.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warn", action="store_true",
                    help="report mismatches but always exit 0")
    a = ap.parse_args()
    return report(check(verbose=True), strict=not a.warn)


if __name__ == "__main__":
    sys.exit(main())
