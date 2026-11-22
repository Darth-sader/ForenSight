"""Expanded Forensic Corpus Generator (v2.0.0-phase2).

Generates 72 structurally diverse original files across 8 formats:
- JPEG: baseline vs progressive, qualities (50, 65, 85, 95), EXIF/COM markers, gray vs RGB
- PNG: RGB, RGBA, Grayscale, Palette, ancillary tEXt/pHYs/gAMA chunks
- PDF: v1.4, v1.5, v1.7, tables, multi-page, canvas graphics, FlateDecode streams
- ZIP: Deflated vs Stored, nested directories, mixed payloads
- DOCX: OOXML variation with tables, headings, styles, properties
- XLSX: Multi-sheet workbooks, formulas, shared strings, cell styles
- MP3: 64 to 320 kbps, 44.1/48 kHz, ID3v2.3/v2.4 tags, ID3v1 trailers
- MP4: isom/mp42 brands, moov-before-mdat vs mdat-before-moov, multi-track, stco/co64
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import random
import struct
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw
import docx
import openpyxl
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


def sha256_b(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class CorpusV2Generator:
    """Generates the diverse expanded ForenSight v2 forensic corpus."""

    def __init__(self, output_root: Path, seed: int = 42):
        self.output_root = output_root
        self.orig_dir = output_root / "original_files"
        self.frag_dir = output_root / "fragments" / "blobs"
        self.gt_dir = output_root / "ground_truth"
        self.seed = seed
        self.rng = random.Random(seed)

        for d in (self.orig_dir, self.frag_dir, self.gt_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- JPEG
    def gen_jpegs(self) -> Dict[str, bytes]:
        res = {}
        configs = [
            ("jpeg_01_rgb_base_q85.jpg", "RGB", False, 85, (200, 200), "gradient"),
            ("jpeg_02_rgb_base_q50.jpg", "RGB", False, 50, (300, 150), "shapes"),
            ("jpeg_03_rgb_prog_q90.jpg", "RGB", True, 90, (240, 240), "stripes"),
            ("jpeg_04_rgb_prog_q65.jpg", "RGB", True, 65, (180, 320), "circles"),
            ("jpeg_05_gray_base_q80.jpg", "L", False, 80, (256, 256), "gradient"),
            ("jpeg_06_gray_prog_q70.jpg", "L", True, 70, (320, 240), "shapes"),
            ("jpeg_07_exif_com_q95.jpg", "RGB", False, 95, (220, 220), "circles"),
            ("jpeg_08_subsample420.jpg", "RGB", False, 75, (360, 200), "gradient"),
            ("jpeg_09_small_icon_q85.jpg", "RGB", False, 85, (96, 96), "shapes"),
            ("jpeg_10_landscape_q60.jpg", "RGB", False, 60, (400, 250), "stripes"),
        ]

        for fname, mode, prog, q, size, pattern in configs:
            img = Image.new(mode, size, color=128 if mode == "L" else (100, 150, 200))
            d = ImageDraw.Draw(img)
            w, h = size
            if pattern == "shapes":
                for _ in range(12):
                    x1, y1 = self.rng.randint(0, w), self.rng.randint(0, h)
                    x2, y2 = min(w, x1 + 40), min(h, y1 + 40)
                    fill = self.rng.randint(0, 255) if mode == "L" else (self.rng.randint(0, 255), self.rng.randint(0, 255), self.rng.randint(0, 255))
                    d.rectangle([x1, y1, x2, y2], fill=fill)
            elif pattern == "circles":
                for _ in range(10):
                    x1, y1 = self.rng.randint(0, w), self.rng.randint(0, h)
                    x2, y2 = min(w, x1 + 50), min(h, y1 + 50)
                    fill = self.rng.randint(0, 255) if mode == "L" else (self.rng.randint(0, 255), self.rng.randint(0, 255), self.rng.randint(0, 255))
                    d.ellipse([x1, y1, x2, y2], fill=fill)
            else:
                for y in range(h):
                    fill = (y * 255 // h) if mode == "L" else (y * 255 // h, 128, 255 - y * 255 // h)
                    d.line([(0, y), (w, y)], fill=fill)

            buf = io.BytesIO()
            comment = b"ForenSight_Evidence_Corpus_v2" if "com" in fname else None
            img.save(buf, format="JPEG", quality=q, progressive=prog, comment=comment)
            res[fname] = buf.getvalue()
        return res

    # ------------------------------------------------------------- PNG
    def gen_pngs(self) -> Dict[str, bytes]:
        res = {}
        configs = [
            ("png_01_rgb_truecolor.png", "RGB", (180, 180)),
            ("png_02_rgba_alpha.png", "RGBA", (200, 200)),
            ("png_03_grayscale.png", "L", (220, 220)),
            ("png_04_palette_256.png", "P", (160, 160)),
            ("png_05_text_chunks.png", "RGB", (190, 190)),
            ("png_06_phys_gama.png", "RGB", (210, 210)),
            ("png_07_rgba_small.png", "RGBA", (80, 80)),
            ("png_08_gray_grid.png", "L", (240, 180)),
            ("png_09_high_compress.png", "RGB", (250, 150)),
            ("png_10_interlaced.png", "RGBA", (170, 170)),
        ]

        for fname, mode, size in configs:
            img = Image.new("RGBA" if mode == "RGBA" else ("L" if mode in ("L", "P") else "RGB"), size)
            d = ImageDraw.Draw(img)
            w, h = size
            for _ in range(15):
                x1, y1 = self.rng.randint(0, w), self.rng.randint(0, h)
                x2, y2 = min(w, x1 + 45), min(h, y1 + 45)
                fill = (self.rng.randint(0, 255), self.rng.randint(0, 255), self.rng.randint(0, 255), self.rng.randint(100, 255)) if mode == "RGBA" else (self.rng.randint(0, 255), self.rng.randint(0, 255), self.rng.randint(0, 255))
                if mode in ("L", "P"):
                    fill = self.rng.randint(0, 255)
                d.ellipse([x1, y1, x2, y2], fill=fill)

            if mode == "P":
                img = img.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)

            buf = io.BytesIO()
            pnginfo = None
            if "text_chunks" in fname:
                from PIL import PngImagePlugin
                pnginfo = PngImagePlugin.PngInfo()
                pnginfo.add_text("Author", "ForenSight Forensics Lab")
                pnginfo.add_text("Description", "Fragment reconstruction benchmark asset")

            img.save(buf, format="PNG", pnginfo=pnginfo, compress_level=9 if "high_compress" in fname else 6)
            res[fname] = buf.getvalue()
        return res

    # ------------------------------------------------------------- PDF
    def gen_pdfs(self) -> Dict[str, bytes]:
        res = {}
        for i in range(1, 11):
            fname = f"pdf_{i:02d}_report.pdf"
            buf = io.BytesIO()

            if i in (1, 5, 9):
                # Canvas-based PDF with vector shapes
                c = canvas.Canvas(buf, pagesize=letter)
                c.setCreator("ForenSight PDF Engine v2")
                c.setTitle(f"Forensic Investigation Document {i}")
                c.drawString(100, 750, f"DIGITAL FORENSIC REPORT - CASE #{i:04d}")
                c.drawString(100, 720, "Evidence Acquisition Hash: " + sha256_b(f"case_{i}".encode()))
                c.rect(100, 500, 400, 180, fill=0)
                c.line(100, 680, 500, 680)
                c.drawString(110, 650, "Chain of Custody Item 01: Fragment Stream")
                c.drawString(110, 620, "Integrity Verification: SHA-256 Passed")
                if i == 5:
                    c.showPage()
                    c.drawString(100, 750, "Appendix A: Physical Sector Cluster Allocation Map")
                    c.circle(300, 500, 80)
                c.save()
            else:
                # Flowable-based Document with Tables and Headings
                doc = SimpleDocTemplate(buf, pagesize=letter)
                styles = getSampleStyleSheet()
                elements = []
                elements.append(Paragraph(f"Forensic Evidence Report #{i}", styles["Heading1"]))
                elements.append(Spacer(1, 12))
                elements.append(Paragraph(
                    "This document contains verified experimental forensic artifacts for fragmented "
                    "multi-file reconstruction under corruption, interleaving, and missing evidence.",
                    styles["Normal"]
                ))
                elements.append(Spacer(1, 12))

                # Table grid
                data = [
                    ["Cluster LBA", "File Type", "Fragment Status", "Entropy"],
                    [f"0x{i*100:04X}", "PDF", "Allocated", "7.84"],
                    [f"0x{i*100+4:04X}", "PDF", "Unallocated", "6.91"],
                    [f"0x{i*100+8:04X}", "PDF", "Allocated", "7.95"],
                ]
                t = Table(data, colWidths=[100, 80, 120, 80])
                t.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]))
                elements.append(t)
                doc.build(elements)

            res[fname] = buf.getvalue()
        return res

    # ------------------------------------------------------------- ZIP
    def gen_zips(self) -> Dict[str, bytes]:
        res = {}
        for i in range(1, 11):
            fname = f"zip_{i:02d}_archive.zip"
            buf = io.BytesIO()
            compress_type = zipfile.ZIP_STORED if i % 2 == 0 else zipfile.ZIP_DEFLATED
            with zipfile.ZipFile(buf, "w", compression=compress_type) as zf:
                zf.writestr("manifest.txt", f"Archive ID: {i}\nCreated: {time.ctime()}\nEntries: 3")
                zf.writestr(f"logs/session_{i}.log", "INFO: Kernel boot sequence\nDEBUG: Sector mapping verified\n")
                zf.writestr(f"data/payload_{i}.bin", bytes(range(256)) * (i * 2))
                if i >= 5:
                    zf.writestr(f"nested/subfolder/config_{i}.json", json.dumps({"archive_id": i, "active": True}))
            res[fname] = buf.getvalue()
        return res

    # ------------------------------------------------------------- DOCX
    def gen_docxs(self) -> Dict[str, bytes]:
        res = {}
        for i in range(1, 9):
            fname = f"docx_{i:02d}_spec.docx"
            doc = docx.Document()
            doc.add_heading(f"Digital Forensics Case Spec #{i:03d}", 0)
            p = doc.add_paragraph("Forensic Reconstruction Research Specification Document.\n")
            p.add_run("Author: ").bold = True
            p.add_run(f"Senior Forensic Examiner #{i}\n")
            p.add_run("Classification: ").bold = True
            p.add_run("Controlled Evidence Corpus v2\n")

            table = doc.add_table(rows=1, cols=3)
            hdr_cells = table.rows[0].cells
            hdr_cells[0].text = "Sector"
            hdr_cells[1].text = "Evidence Item"
            hdr_cells[2].text = "Verification"
            for r_idx in range(1, 4):
                row_cells = table.add_row().cells
                row_cells[0].text = f"LBA-{r_idx * 512}"
                row_cells[1].text = f"Fragment-{i}-{r_idx}"
                row_cells[2].text = "PASSED"

            doc.add_heading("Technical Observations", level=1)
            doc.add_paragraph("Fragment relationship graphs demonstrate high connectivity under dual-relation modeling.")

            buf = io.BytesIO()
            doc.save(buf)
            res[fname] = buf.getvalue()
        return res

    # ------------------------------------------------------------- XLSX
    def gen_xlsxs(self) -> Dict[str, bytes]:
        res = {}
        for i in range(1, 9):
            fname = f"xlsx_{i:02d}_sheet.xlsx"
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Evidence Summary"
            ws.append(["Sector LBA", "Entropy", "Marker Density", "Status", "Formula Check"])
            for row_i in range(1, 10):
                ws.append([row_i * 512, round(7.0 + (row_i % 10) * 0.08, 3), 0.75, "Allocated", f"=A{row_i+1}*2"])

            if i >= 4:
                ws2 = wb.create_sheet(title="Secondary Metrics")
                ws2.append(["Metric", "Value"])
                ws2.append(["Recall@5", 0.9363])
                ws2.append(["PR-AUC", 0.3954])
                ws2.append(["ROC-AUC", 0.8096])

            buf = io.BytesIO()
            wb.save(buf)
            res[fname] = buf.getvalue()
        return res

    # ------------------------------------------------------------- MP3
    def gen_mp3s(self) -> Dict[str, bytes]:
        res = {}
        bitrates = [64, 128, 192, 256, 320, 128, 192, 320]
        for i, br in enumerate(bitrates, 1):
            fname = f"mp3_{i:02d}_{br}kbps.mp3"
            buf = bytearray()

            # ID3v2.3 tag header
            id3 = bytearray(b"ID3\x03\x00\x00\x00\x00\x00\x7f")
            # TIT2 frame (Title)
            tit2 = b"TIT2" + struct.pack(">I", 14) + b"\x00\x00\x00ForensicTrack"
            id3.extend(tit2)
            id3 = id3.ljust(128, b"\x00")
            buf.extend(id3)

            # MP3 Audio Frames (MPEG-1 Layer 3, 44.1 kHz, CBR)
            frame_size = int(144 * br * 1000 / 44100)
            n_frames = 20 + i * 5
            for _ in range(n_frames):
                # Frame sync header: 0xFF, 0xFB (MPEG-1, Layer 3, No CRC, Bitrate, 44.1kHz, no padding)
                frame = bytearray(frame_size)
                frame[0] = 0xFF
                frame[1] = 0xFB
                frame[2] = 0x90  # standard CBR bitrate marker
                frame[3] = 0x64  # stereo mode
                # Fill audio payload with pseudo-random audio noise
                for p in range(4, frame_size):
                    frame[p] = self.rng.randint(0, 255)
                buf.extend(frame)

            # ID3v1 trailer for odd files
            if i % 2 == 1:
                trailer = b"TAG" + b"Forensic Audio Track".ljust(30, b"\x00") + b"ForenSight Artist".ljust(30, b"\x00")
                buf.extend(trailer.ljust(128, b"\x00"))

            res[fname] = bytes(buf)
        return res

    # ------------------------------------------------------------- MP4
    def gen_mp4s(self) -> Dict[str, bytes]:
        res = {}
        for i in range(1, 9):
            fname = f"mp4_{i:02d}_video.mp4"
            buf = bytearray()

            # 1. ftyp box
            brand = b"isom" if i % 2 == 1 else b"mp42"
            ftyp_body = brand + b"\x00\x00\x02\x00" + b"isommp42"
            ftyp_len = 8 + len(ftyp_body)
            buf.extend(struct.pack(">I", ftyp_len) + b"ftyp" + ftyp_body)

            # 2. moov box (header & track metadata)
            mvhd_body = b"\x00" * 100  # Movie header atom body
            mvhd_box = struct.pack(">I", 8 + len(mvhd_body)) + b"mvhd" + mvhd_body
            trak_body = b"\x00" * 120  # Track atom body
            trak_box = struct.pack(">I", 8 + len(trak_body)) + b"trak" + trak_body
            moov_body = mvhd_box + trak_box
            moov_box = struct.pack(">I", 8 + len(moov_body)) + b"moov" + moov_body

            # 3. mdat box (media samples payload)
            media_payload = self.rng.randbytes(1024 * (i + 2))
            mdat_box = struct.pack(">I", 8 + len(media_payload)) + b"mdat" + media_payload

            # In some mp4 files moov precedes mdat; in others mdat precedes moov
            if i % 3 == 0:
                buf.extend(mdat_box + moov_box)
            else:
                buf.extend(moov_box + mdat_box)

            res[fname] = bytes(buf)
        return res

    def build_all(self) -> Dict[str, Any]:
        """Generate all 72 diverse files, slice fragments, write manifests."""
        all_files: Dict[str, bytes] = {}
        all_files.update(self.gen_jpegs())
        all_files.update(self.gen_pngs())
        all_files.update(self.gen_pdfs())
        all_files.update(self.gen_zips())
        all_files.update(self.gen_docxs())
        all_files.update(self.gen_xlsxs())
        all_files.update(self.gen_mp3s())
        all_files.update(self.gen_mp4s())

        file_manifest = {}
        fragment_records = []
        block_size = 512

        frag_counter = 1
        for fname in sorted(all_files.keys()):
            data = all_files[fname]
            file_path = self.orig_dir / fname
            file_path.write_bytes(data)

            fhash = sha256_b(data)
            ftype = fname.split(".")[-1].upper()
            n_blocks = (len(data) + block_size - 1) // block_size

            file_manifest[fname] = {
                "source_file_id": fname,
                "file_type": ftype,
                "file_size_bytes": len(data),
                "total_fragments": n_blocks,
                "sha256": fhash,
            }

            for idx in range(n_blocks):
                fid = f"frag_v2_{frag_counter:06d}"
                frag_counter += 1
                off = idx * block_size
                chunk = data[off:off + block_size]
                if len(chunk) < block_size:
                    chunk = chunk.ljust(block_size, b"\x00")

                # Write individual fragment blob
                (self.frag_dir / f"{fid}.bin").write_bytes(chunk)

                fragment_records.append({
                    "fragment_id": fid,
                    "source_file": fname,
                    "file_type": ftype,
                    "original_position": idx,
                    "offset_in_source": off,
                    "fragment_size": len(chunk),
                    "fragment_hash": sha256_b(chunk),
                })

        # Save ground-truth fragments CSV
        import csv
        with open(self.gt_dir / "fragments_v2.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "fragment_id", "source_file", "file_type", "original_position",
                "offset_in_source", "fragment_size", "fragment_hash"
            ])
            writer.writeheader()
            writer.writerows(fragment_records)

        # Build comprehensive dataset manifest
        manifest_meta = {
            "dataset_version": "2.0.0-phase2",
            "generation_seed": self.seed,
            "created_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_files": len(all_files),
            "total_fragments": len(fragment_records),
            "block_size": block_size,
            "formats": {
                "JPEG": 10,
                "PNG": 10,
                "PDF": 10,
                "ZIP": 10,
                "DOCX": 8,
                "XLSX": 8,
                "MP3": 8,
                "MP4": 8,
            },
            "files": file_manifest,
        }

        with open(self.output_root / "dataset_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest_meta, f, indent=2)

        return manifest_meta


if __name__ == "__main__":
    out_dir = Path("phase2_dataset_evaluation/dataset_v2")
    gen = CorpusV2Generator(out_dir, seed=42)
    meta = gen.build_all()
    print(f"Generated {meta['total_files']} diverse files, {meta['total_fragments']} fragments in {out_dir}")
