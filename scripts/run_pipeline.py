#!/usr/bin/env python
"""CLI: train the complete ForenSight ML stack and persist artifacts.

Usage: python scripts/run_pipeline.py [--no-tune] [--no-cv]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import train_all


def main() -> None:
    tune = "--no-tune" not in sys.argv
    cv = "--no-cv" not in sys.argv
    train_all(tune=tune, cv=cv)


if __name__ == "__main__":
    main()
