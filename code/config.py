"""Central configuration for FAITHDROID.

All paths, split definitions and hyper-parameters live here so that every experiment
is reproducible from a single source of truth.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "kronodroid" / "extracted"
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
IMAGES = ROOT / "images"
CACHE = ROOT / "data" / "llm_cache"
for _d in (PROC, RESULTS, IMAGES, CACHE):
    _d.mkdir(parents=True, exist_ok=True)

MALWARE_CSV = RAW / "real_malware_v1.csv"
BENIGN_CSV = RAW / "real_legitimate_v1.csv"

SEED = 42

# ---------------------------------------------------------------- temporal splits
# KronoDroid's HighestModDate is 98.2% usable (76,716/78,137 rows); EarliestModDate is
# only 88.5% usable because 6,897 rows sit at the ZIP epoch 1980-01-01.
TIME_COL = "HighestModDate"
TIME_MIN = "2008-01-01"   # anything earlier is a ZIP-epoch / corrupt artefact
TIME_MAX = "2021-06-30"   # dataset was published mid-2021

# Train on everything up to and including TRAIN_END; three later windows probe
# incremental drift, mirroring LAMD's Test 1/2/3 design.
TRAIN_END = 2014
TEST_WINDOWS = {
    "Test1": (2015, 2016),
    "Test2": (2017, 2018),
    "Test3": (2019, 2020),
}

# KronoDroid's per-year class balance swings violently (2011 is 88% benign; 2012-14 is
# ~94% malware). Left alone, a temporal split confounds concept drift with prior shift.
# We therefore pin malware prevalence in every evaluation window to match LAMD's
# 9.42-10.01% so that our numbers are comparable to theirs and measure drift, not prior.
TARGET_PREVALENCE = 0.10

# Fraction of the training window held back to calibrate the conformal predictor.
#
# NOTE: this is a *stratified random* split within the training period, not a
# chronological sub-split. A chronological sub-split is degenerate on KronoDroid:
# benign samples concentrate in 2010-2011 and malware in 2012-2014, so cutting the
# training pool by time yields 1.0% malware in one half and 31.0% in the other,
# which destroys both training signal and calibration validity. A stratified split
# keeps both halves at TARGET_PREVALENCE while remaining strictly time-anterior to
# every test window - the property the drift experiment actually requires.
CALIB_FRACTION = 0.30

# Unnamed system-call columns (SYS_300..SYS_369) are raw syscall numbers with no
# symbolic name. They are legitimate classifier features but are excluded from the
# LLM evidence bundle: "SYS_342" carries no meaning a language model can reason
# about, and citing it would be uninformative to an analyst.
UNNAMED_SYSCALL_PREFIX = "SYS_"

# ---------------------------------------------------------------- feature groups
# Columns that are identifiers, labels or leakage risks - never used as features.
META_COLS = [
    "Package", "sha256", "MalFamily", "Malware",
    "EarliestModDate", "HighestModDate",
    # Leakage: these are VirusTotal-derived, i.e. they encode the label itself.
    "Detection_Ratio", "Scanners", "TimesSubmitted",
]

# Aggregate counters that are neither a named permission nor a named syscall.
AGG_COLS = [
    "nr_syscalls", "nr_permissions", "nr_custom", "total_perm",
    "normal", "dangerous", "signature", "custom_yes",
    "CFileSize", "UFileSize", "FilesInsideAPK", "NrContactedIps",
    "Activities", "NrIntServices", "NrIntServicesActions",
    "NrIntActivities", "NrIntActivitiesActions",
    "NrIntReceivers", "NrIntReceiversActions",
    "TotalIntentFilters", "NrServices",
]

# ---------------------------------------------------------------- gate
# Nominal significance for the Mondrian conformal predictor. Retained for
# validity-backed ABSTENTION, not for routing - see docs/02_findings_log.md F3.
CONFORMAL_EPSILON = 0.10

# Routing signal. The conformal escalation score was measured to be anti-selective
# under drift (mean lift 0.49-0.78, i.e. at or below chance) because 36-45% of test
# samples flag as novel and the signal saturates. Probability-margin uncertainty
# weighted toward the positive class achieves mean lift ~13x instead.
ROUTING_SIGNAL = "margin_x_malware"

# Escalation budgets swept to build the cost-accuracy Pareto curve.
ESCALATION_BUDGETS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00]

# ---------------------------------------------------------------- LLM stage
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_PRIMARY = "qwen2.5:7b"
LLM_SWEEP = ["llama3.2:3b", "qwen2.5:7b", "phi4:14b"]
LLM_TIMEOUT = 300
# Concurrent requests to Ollama. Measured ~20% throughput gain at 6 workers; the
# server has a single slot so this mainly overlaps prompt eval with generation.
LLM_WORKERS = 6
LLM_TEMPERATURE = 0.0
LLM_NUM_PREDICT = 420
# How many artifacts of each kind to show the model in the evidence bundle.
BUNDLE_TOP_PERMS = 25
BUNDLE_TOP_SYSCALLS = 25

# ---------------------------------------------------------------- faithfulness
# An explanation citing fewer than this many grounded artifacts is treated as
# unsupported regardless of its EGR.
MIN_CITATIONS = 1
# Abstain when the grounding rate of an explanation falls below this.
EGR_ABSTAIN_THRESHOLD = 0.75
