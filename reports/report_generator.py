"""Forensic evaluation report generator (doc §36): ranked, interpretable,
evidence-supported reconstruction candidates exported as JSON + HTML + text."""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from common import ReconstructionCandidate, save_json
from config import REPORTS_DIR


def candidate_to_text(c: ReconstructionCandidate, rank: int) -> str:
    lines = [
        f"RECOVERY CANDIDATE #{rank}",
        f"File Type: {c.file_type}",
        f"Estimated Size: {c.estimated_size / 1024:.1f} KB ({c.estimated_size} bytes)",
        f"Fragment Sequence: {c.sequence_string()}",
        f"Calibrated Confidence: {c.confidence * 100:.1f}%",
    ]
    lines.append("ML Evidence:")
    for k, v in c.ml_evidence.items():
        lines.append(f"  + {k}: {v}")
    lines.append("Forensic Evidence:")
    fe = c.forensic_evidence or {}
    for k in ("header_valid", "footer_valid", "no_impossible_reuse",
              "filesystem_consistent", "size_in_typical_range"):
        if k in fe:
            mark = "+" if fe[k] else "-"
            lines.append(f"  {mark} {k}: {fe[k]}")
    lines.append("Anomaly Analysis:")
    aa = c.anomaly_analysis or {}
    lines.append(f"  max anomaly score: {aa.get('max_anomaly', 0)}")
    for f in aa.get("flagged", []):
        lines.append(f"  ! {f}")
    if c.issues:
        lines.append("Issues:")
        for i in c.issues:
            lines.append(f"  ! {i}")
    lines.append(f"Classification: {c.classification}")
    lines.append("")
    return "\n".join(lines)


def generate_reports(scenario_id: str, image_name: str,
                     candidates: List[ReconstructionCandidate],
                     metrics: Optional[Dict[str, object]] = None,
                     summary: Optional[Dict[str, object]] = None) -> Dict[str, str]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"forensight_report_{scenario_id}_{stamp}"

    # ------------------------------------------------------------------ text
    text = "\n".join([
        "=" * 72,
        "ForenSight - Forensic Evaluation Report",
        "=" * 72,
        f"Generated:    {datetime.now().isoformat(timespec='seconds')}",
        f"Scenario:     {scenario_id}",
        f"Disk image:   {image_name}",
        f"Candidates:   {len(candidates)}",
        "",
    ])
    if summary:
        text += "ANALYSIS SUMMARY\n" + "-" * 40 + "\n"
        for k, v in summary.items():
            text += f"{k}: {v}\n"
        text += "\n"
    if metrics:
        text += "QUANTITATIVE METRICS\n" + "-" * 40 + "\n"
        for k, v in metrics.items():
            text += f"{k}: {v}\n"
        text += "\n"
    for i, c in enumerate(candidates, 1):
        text += candidate_to_text(c, i)
    text += "=" * 72 + "\nML proposes. Constraints validate. Predictions are not\n" \
                     "automatically treated as forensic truth.\n" + "=" * 72 + "\n"
    txt_path = REPORTS_DIR / f"{base}.txt"
    txt_path.write_text(text, encoding="utf-8")

    # ------------------------------------------------------------------ json
    json_path = REPORTS_DIR / f"{base}.json"
    save_json(json_path, {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "scenario": scenario_id, "disk_image": image_name,
        "metrics": metrics or {}, "summary": summary or {},
        "candidates": [c.__dict__ for c in candidates],
    })

    # ------------------------------------------------------------------ html
    rows = []
    for i, c in enumerate(candidates, 1):
        cls_color = ("#2e7d32" if c.classification == "HIGH-CONFIDENCE RECOVERY"
                     else "#ef6c00" if c.classification == "UNCERTAIN" else "#c62828")
        issues = "".join(f"<li>{html.escape(x)}</li>" for x in c.issues) or "<li>none</li>"
        rows.append(f"""
        <div class="cand">
          <h3>RECOVERY CANDIDATE #{i} <span class="tag" style="background:{cls_color}">{c.classification}</span></h3>
          <p><b>File Type:</b> {c.file_type} &nbsp; <b>Estimated Size:</b> {c.estimated_size/1024:.1f} KB
             &nbsp; <b>Calibrated Confidence:</b> {c.confidence*100:.1f}%</p>
          <p class="seq"><b>Fragment Sequence:</b> {html.escape(c.sequence_string())}</p>
          <p><b>ML Evidence:</b> transition={c.ml_evidence.get('transition_mean')},
             embedding={c.ml_evidence.get('embedding_similarity')},
             structural={c.ml_evidence.get('structural_consistency')}</p>
          <p><b>Anomaly:</b> max={c.anomaly_analysis.get('max_anomaly', 0)}
             flagged={len(c.anomaly_analysis.get('flagged', []))}</p>
          <details><summary>Issues</summary><ul>{issues}</ul></details>
        </div>""")
    metrics_html = "".join(f"<tr><td>{html.escape(str(k))}</td><td>{v}</td></tr>"
                           for k, v in (metrics or {}).items())
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>ForenSight Report - {scenario_id}</title><style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#f5f6f8;color:#222}}
h1{{font-size:22px}} .cand{{background:#fff;border:1px solid #ddd;border-radius:8px;
padding:14px 18px;margin:14px 0}} .tag{{color:#fff;font-size:11px;padding:2px 8px;
border-radius:10px;vertical-align:middle}} .seq{{font-family:Consolas,monospace;
font-size:12px;background:#f0f3f7;padding:6px 8px;border-radius:4px}}
table{{border-collapse:collapse}} td{{border:1px solid #ccc;padding:4px 10px;font-size:13px}}
</style></head><body>
<h1>ForenSight &mdash; Forensic Evaluation Report</h1>
<p>Scenario <b>{html.escape(scenario_id)}</b> &middot; image <b>{html.escape(image_name)}</b>
&middot; generated {datetime.now().isoformat(timespec='seconds')}</p>
<h2>Metrics</h2><table>{metrics_html}</table>
<h2>Ranked Recovery Candidates ({len(candidates)})</h2>
{''.join(rows)}
<p style="color:#666;font-size:11px">ML proposes. Constraints validate. Predictions are not
automatically treated as forensic truth.</p>
</body></html>"""
    html_path = REPORTS_DIR / f"{base}.html"
    html_path.write_text(doc, encoding="utf-8")

    return {"text": str(txt_path), "json": str(json_path), "html": str(html_path)}
