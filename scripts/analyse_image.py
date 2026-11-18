#!/usr/bin/env python
"""CLI: analyse a disk image with the trained ForenSight stack.

Usage: python scripts/analyse_image.py dataset/disk_images/S_H.img [name]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import analyse_image
from reports import generate_reports


def main() -> None:
    image = Path(sys.argv[1])
    name = sys.argv[2] if len(sys.argv) > 2 else image.stem
    result = analyse_image(image, name)
    cands = result["candidates"]
    print(f"\n=== {name}: {len(cands)} ranked candidates "
          f"(mode={result['reconstruction_mode']}, {result['seconds']}s) ===")
    for i, c in enumerate(cands[:10], 1):
        print(f"#{i:02d} [{c.classification}] conf={c.confidence:.3f} type={c.file_type} "
              f"len={len(c.fragment_sequence)} {c.sequence_string()[:90]}")
    if result["metrics"]:
        import json
        print("\nmetrics:", json.dumps(result["metrics"], indent=1, default=str)[:1200])
    reports.generate_reports(name, image.name, cands, result["metrics"],
                             {"fragments": len(result["fragments"]),
                              "graph_edges": len(result["graph"]["edges"]),
                              "anomalies": result["anomaly"]["flagged"],
                              "clusters": result["clusters"]["k"]})


if __name__ == "__main__":
    main()
