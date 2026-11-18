#!/usr/bin/env python
"""CLI: generate the controlled forensic corpus and scenario disk images (doc §6-7)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import set_seeds
from config import SEED
from extraction.dataset_generator import generate_all


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else SEED
    set_seeds(seed)
    generate_all(seed)


if __name__ == "__main__":
    main()
