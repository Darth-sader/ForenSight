"""ForenSight global configuration.

All paths are anchored at the project root so every artifact stays inside
D:\\Projects\\ForenSight regardless of the working directory.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- directories
DATASET_DIR = PROJECT_ROOT / "dataset"
ORIGINAL_FILES_DIR = DATASET_DIR / "original_files"
DISK_IMAGES_DIR = DATASET_DIR / "disk_images"
FRAGMENTS_DIR = DATASET_DIR / "fragments"
GROUND_TRUTH_DIR = DATASET_DIR / "ground_truth"
SCENARIOS_DIR = DATASET_DIR / "scenarios"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"          # trained models, encoders
RESULTS_DIR = PROJECT_ROOT / "results"              # metrics + experiment JSON
PLOTS_DIR = RESULTS_DIR / "plots"                   # PNG visualizations
REPORTS_DIR = PROJECT_ROOT / "reports" / "generated"
LOGS_DIR = PROJECT_ROOT / "logs"
FRONTEND_DEMO_DIR = PROJECT_ROOT / "frontend" / "public" / "demo"

for _d in (DATASET_DIR, ORIGINAL_FILES_DIR, DISK_IMAGES_DIR, FRAGMENTS_DIR,
           GROUND_TRUTH_DIR, SCENARIOS_DIR, ARTIFACTS_DIR, RESULTS_DIR,
           PLOTS_DIR, REPORTS_DIR, LOGS_DIR, FRONTEND_DEMO_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- reproducibility
SEED = int(os.environ.get("FORENSIGHT_SEED", "42"))

# ---------------------------------------------------------------- corpus
FRAGMENT_SIZE = int(os.environ.get("FORENSIGHT_FRAGMENT_SIZE", "512"))
DISK_BLOCK_SIZE = FRAGMENT_SIZE          # simulated FS uses one cluster per fragment
SUPERBLOCK_BLOCKS = 8                    # header region (superblock + allocation bitmap)

# how many original files of each type to synthesise
CORPUS_PLAN = {
    "JPEG": 10,
    "PNG": 8,
    "PDF": 8,
    "ZIP": 6,
    "DOCX": 6,
    "XLSX": 4,
    "MP3": 4,
    "MP4": 4,
}

# fraction of files reserved for generalisation testing (unseen files, doc §27)
HOLDOUT_FILE_FRACTION = 0.25

# ---------------------------------------------------------------- scenarios
CORRUPTION_LEVELS = [0.01, 0.05, 0.10, 0.20]
MISSING_LEVELS = [0.05, 0.10, 0.20]
FRAGMENTATION_LEVELS = ["contiguous", "low", "medium", "high", "severe"]
# interleave factor: how many foreign blocks are inserted between consecutive
# fragments of a file for each fragmentation level
FRAGMENTATION_INTERLEAVE = {"contiguous": 0, "low": 1, "medium": 3, "high": 6, "severe": 12}

# ---------------------------------------------------------------- ML pipeline
PAIR_NEGATIVE_RATIO = 2.0        # negatives per positive (next-fragment task)
SAME_PAIR_RATIO = 1.0            # same-file non-adjacent pairs per consecutive pair
HARD_NEGATIVE_FRACTION = 0.5     # fraction of negatives that are same-type/hard

EMBEDDING_DIM = 32
SIAMESE_MARGIN = 1.0
TORCH_THREADS = 4

GRAPH_TOP_K_SUCCESSORS = 5       # edges kept per fragment in relationship graph
BEAM_WIDTH = 6
MAX_CANDIDATES_PER_SEED = 5
MAX_SEQUENCE_LENGTH = 64         # hard cap on candidate reconstruction length

# combined reconstruction score weights (empirically calibrated, doc §20)
SCORE_WEIGHTS = {
    "transition": 0.40,
    "embedding": 0.20,
    "structural": 0.20,
    "context": 0.10,
    "anomaly_penalty": 0.10,
    "violation_penalty": 0.35,
}
GAP_CONFIDENCE_THRESHOLD = 0.35  # below this a [UNKNOWN] gap is emitted (doc §31)

CONFIDENCE_BANDS = {"HIGH-CONFIDENCE RECOVERY": 0.85, "UNCERTAIN": 0.60}
# below UNCERTAIN threshold -> "LOW-CONFIDENCE RECOVERY"

# ---------------------------------------------------------------- database / api
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "sqlite:///" + str(PROJECT_ROOT / "forensight.db"),
)
API_HOST = os.environ.get("FORENSIGHT_API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("FORENSIGHT_API_PORT", "8000"))
API_CORS_ORIGINS = os.environ.get(
    "FORENSIGHT_CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
).split(",")


def ensure_dirs() -> None:
    for _d in (DATASET_DIR, ORIGINAL_FILES_DIR, DISK_IMAGES_DIR, FRAGMENTS_DIR,
               GROUND_TRUTH_DIR, SCENARIOS_DIR, ARTIFACTS_DIR, RESULTS_DIR,
               PLOTS_DIR, REPORTS_DIR, LOGS_DIR, FRONTEND_DEMO_DIR):
        _d.mkdir(parents=True, exist_ok=True)
