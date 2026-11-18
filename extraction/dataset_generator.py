"""Controlled forensic corpus generation (doc §6-7).

Synthesises original files with known ground truth (JPEG, PNG, PDF, ZIP, DOCX,
XLSX, MP3, MP4), splits them into fragments and lays the fragments out on
simulated disk images under scenarios A-H (contiguous, fragmentation,
corruption, missing fragments, mixed files, similar content, adversarial).
"""
from __future__ import annotations

import csv
import random
import struct
import zipfile
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from common import logger, save_json, set_seeds, timer
from config import (CORPUS_PLAN, CORRUPTION_LEVELS, DISK_BLOCK_SIZE,
                    FRAGMENTATION_INTERLEAVE, FRAGMENT_SIZE,
                    GROUND_TRUTH_DIR, HOLDOUT_FILE_FRACTION, MISSING_LEVELS,
                    ORIGINAL_FILES_DIR, DISK_IMAGES_DIR, SCENARIOS_DIR,
                    FRAGMENTS_DIR, SEED, SUPERBLOCK_BLOCKS)

FSIM_MAGIC = b"FSIM0001"

# --------------------------------------------------------------------------
#  Original file synthesis
# --------------------------------------------------------------------------

_WORDS = ("forensic evidence fragment disk recovery image analysis report data "
          "sector cluster file system scan detect reconstruct model training "
          "feature vector embedding confidence candidate anomaly graph").split()


def _random_text(rng: random.Random, n_words: int) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(n_words))


def make_jpeg(rng: random.Random, target_bytes: int, base_image=None):
    from PIL import Image, ImageDraw

    if base_image is not None:
        img = base_image.copy()
    else:
        w, h = 240, 240
        img = Image.new("RGB", (w, h))
        px = img.load()
        r0, g0, b0 = rng.randrange(256), rng.randrange(256), rng.randrange(256)
        for y in range(h):
            for x in range(w):
                px[x, y] = ((r0 + x) % 256, (g0 + y) % 256, (b0 + x * y // 256) % 256)
    d = ImageDraw.Draw(img)
    for _ in range(rng.randint(6, 14)):
        x1, y1 = rng.randrange(0, img.width), rng.randrange(0, img.height)
        x2, y2 = min(img.width, x1 + rng.randint(10, 90)), min(img.height, y1 + rng.randint(10, 90))
        d.ellipse([x1, y1, x2, y2], fill=(rng.randrange(256), rng.randrange(256), rng.randrange(256)))
    if base_image is not None:  # similar-content variant: small perturbation
        for _ in range(3):
            x1, y1 = rng.randrange(0, img.width - 20), rng.randrange(0, img.height - 20)
            d.rectangle([x1, y1, x1 + 15, y1 + 15], fill=(rng.randrange(256),) * 3)
    buf = io.BytesIO()
    quality = 85
    img.save(buf, format="JPEG", quality=quality)
    while buf.tell() < target_bytes and quality < 97:
        quality += 4
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def make_png(rng: random.Random, target_bytes: int):
    from PIL import Image, ImageDraw

    w, h = rng.randint(96, 220), rng.randint(96, 220)
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        base = (y * 7) % 256
        for x in range(w):
            px[x, y] = (base, (base + x) % 256, rng.randrange(200, 256))
    d = ImageDraw.Draw(img)
    for _ in range(rng.randint(4, 10)):
        d.line([rng.randrange(w), rng.randrange(h), rng.randrange(w), rng.randrange(h)],
               fill=(rng.randrange(256),) * 3, width=rng.randint(1, 3))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_pdf(rng: random.Random, target_bytes: int) -> bytes:
    n_pages = max(1, target_bytes // 9000)
    words = [_random_text(rng, rng.randint(40, 120)) for _ in range(max(12, target_bytes // 550))]
    per_page = max(1, len(words) // n_pages)
    objects: List[str] = []
    page_ids = []
    obj_n = 3
    for p in range(n_pages):
        content_words = words[p * per_page:(p + 1) * per_page] or words
        lines = []
        for i in range(0, len(content_words), 12):
            chunk = " ".join(content_words[i:i + 12]).replace("(", "").replace(")", "")
            lines.append(f"BT /F1 11 Tf 72 {740 - (i // 12) * 16 % 700} Td ({chunk}) Tj ET")
        stream = "\n".join(lines)
        content_id = obj_n
        page_id = obj_n + 1
        obj_n += 2
        page_ids.append(page_id)
        objects.append((content_id, f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"))
        objects.append((page_id, f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                                 f"/Contents {content_id} 0 R /Resources << /Font << /F1 5 0 R >> >>"))
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    out = io.StringIO()
    out.write("%PDF-1.4\n")
    out.write("1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
    out.write(f"2 0 obj << /Type /Pages /Kids [{kids}] /Count {n_pages} >> endobj\n")
    offsets = {}
    body_pos = out.tell()
    for oid, body in objects:
        offsets[oid] = body_pos + out.tell()
        out.write(f"{oid} 0 obj {body} endobj\n")
    font_offset = body_pos + out.tell()
    out.write("5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n")
    xref_pos = body_pos + out.tell()
    max_obj = max([*offsets.keys(), 5])
    out.write(f"xref\n0 {max_obj + 1}\n0000000000 65535 f \n")
    for oid in range(1, max_obj + 1):
        off = offsets.get(oid, font_offset)
        out.write(f"{off:010d} 00000 n \n")
    out.write(f"trailer << /Size {max_obj + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n")
    return out.getvalue().encode("latin-1")


def make_zip(rng: random.Random, target_bytes: int) -> bytes:
    buf = io.BytesIO()
    n_entries = rng.randint(4, 8)
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i in range(n_entries):
            if rng.random() < 0.5:
                rows = ["col1,col2,col3"] + [
                    f"{rng.random():.4f},{rng.randint(0, 9999)},{_random_text(rng, 5)}"
                    for _ in range(rng.randint(40, 400))]
                zf.writestr(f"data_{i}.csv", "\n".join(rows))
            else:
                zf.writestr(f"notes_{i}.txt", (_random_text(rng, 30) + "\n") * rng.randint(20, 200))
        zf.writestr("blob.bin", rng.randbytes(max(64, target_bytes // (4 * n_entries))))
    return buf.getvalue()


_DOCX_CT = ("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
            "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
            "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
            "<Override PartName=\"/word/document.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/>"
            "</Types>")
_RELS = ("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
         "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
         "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"word/document.xml\"/>"
         "</Relationships>")


def make_docx(rng: random.Random, target_bytes: int) -> bytes:
    paras = []
    for i in range(max(20, target_bytes // 220)):
        paras.append(f"<w:p><w:r><w:t>{_random_text(rng, rng.randint(8, 24))}</w:t></w:r></w:p>")
    document = ("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
                "<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\">"
                "<w:body>" + "".join(paras) + "</w:body></w:document>")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _DOCX_CT)
        zf.writestr("_rels/.rels", _RELS)
        zf.writestr("word/document.xml", document)
    return buf.getvalue()


def make_xlsx(rng: random.Random, target_bytes: int) -> bytes:
    rows = []
    for r in range(max(40, target_bytes // 130)):
        cells = "".join(f"<c r=\"{c}{r}\"><v>{rng.random():.6f}</v></c>" for c in "ABCDEF")
        rows.append(f"<row r=\"{r}\">{cells}</row>")
    sheet = ("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
             "<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">"
             "<sheetData>" + "".join(rows) + "</sheetData></worksheet>")
    workbook = ("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
                "<workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" "
                "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">"
                "<sheets><sheet name=\"Data\" sheetId=\"1\" r:id=\"rId1\"/></sheets></workbook>")
    wb_rels = ("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
               "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
               "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\" Target=\"worksheets/sheet1.xml\"/>"
               "</Relationships>")
    ct = _DOCX_CT.replace("/word/document.xml", "/xl/workbook.xml").replace(
        "wordprocessingml.document.main", "spreadsheetml.sheet.main")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", ct)
        zf.writestr("_rels/.rels", _RELS)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", wb_rels)
        zf.writestr("xl/worksheets/sheet1.xml", sheet)
    return buf.getvalue()


def make_mp3(rng: random.Random, target_bytes: int) -> bytes:
    # ID3v2 tag
    title = _random_text(rng, 3).encode()
    tag_payload = b"TIT2" + struct.pack(">I", len(title)) + b"\x00\x00" + title
    id3 = b"ID3" + b"\x03\x00" + b"\x00" + struct.pack(">I", len(tag_payload)) + tag_payload
    # MPEG-1 Layer III frames, 128 kbps / 44.1 kHz -> 417 byte frames (+ padding)
    frames = bytearray()
    frame_size = 417
    t = np.arange(frame_size, dtype=np.float64)
    while len(frames) < target_bytes:
        phase = rng.random() * 6.28
        tone = (np.sin(t / 7.0 + phase) * 60 + np.sin(t / 13.0) * 30 + 128).astype(np.uint8)
        frames += b"\xff\xfb\x90\x00" + tone.tobytes()
    return id3 + bytes(frames[: max(0, target_bytes - len(id3))])


def make_mp4(rng: random.Random, target_bytes: int) -> bytes:
    def box(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", 8 + len(payload)) + kind + payload

    ftyp = box(b"ftyp", b"isom\x00\x00\x02\x00isomiso2avc1mp41")
    mdat_chunks = bytearray()
    while len(mdat_chunks) < target_bytes:
        nalu = b"\x00\x00\x00\x01\x41\x9a" + rng.randbytes(rng.randint(256, 1024))
        mdat_chunks += nalu
    mdat = box(b"mdat", bytes(mdat_chunks[: max(0, target_bytes)]))
    mvhd = box(b"mvhd", struct.pack(">BIxxxIIIII", 0, 1000, 1000, 1000, 0x00010000, 0, 2))
    trak = box(b"trak", box(b"tkhd", b"\x00" * 80) + box(b"mdia", box(b"mdhd", b"\x00" * 20)))
    moov = box(b"moov", mvhd + trak)
    return ftyp + mdat + moov


SYNTHESIZERS = {
    "JPEG": lambda rng, target, base: make_jpeg(rng, target, base),
    "PNG": lambda rng, target, base: make_png(rng, target),
    "PDF": lambda rng, target, base: make_pdf(rng, target),
    "ZIP": lambda rng, target, base: make_zip(rng, target),
    "DOCX": lambda rng, target, base: make_docx(rng, target),
    "XLSX": lambda rng, target, base: make_xlsx(rng, target),
    "MP3": lambda rng, target, base: make_mp3(rng, target),
    "MP4": lambda rng, target, base: make_mp4(rng, target),
}

TYPE_TARGETS = {"JPEG": (40_000, 150_000), "PNG": (30_000, 110_000), "PDF": (16_000, 70_000),
                "ZIP": (10_000, 55_000), "DOCX": (9_000, 36_000), "XLSX": (7_000, 28_000),
                "MP3": (55_000, 110_000), "MP4": (80_000, 150_000)}
TYPE_PREFIX = {"JPEG": "photo", "PNG": "scan", "PDF": "report", "ZIP": "archive",
               "DOCX": "doc", "XLSX": "sheet", "MP3": "audio", "MP4": "video"}


# --------------------------------------------------------------------------
#  Corpus builder
# --------------------------------------------------------------------------

class CorpusBuilder:
    """Generates original files, canonical fragments and ground truth (doc §6)."""

    def __init__(self, seed: int = SEED):
        self.rng = random.Random(seed)
        self.seed = seed
        self.files: List[Dict[str, object]] = []
        self.fragments: List[Dict[str, object]] = []   # metadata rows
        self.blobs: Dict[str, bytes] = {}              # fragment_id -> bytes
        self._jpeg_bases: Dict[int, object] = {}

    # -------------------------------------------------------------- files
    def generate_files(self) -> List[Dict[str, object]]:
        idx = 1
        for ftype, count in CORPUS_PLAN.items():
            lo, hi = TYPE_TARGETS[ftype]
            for k in range(count):
                # scenario G: even-numbered JPEGs/PDFs reuse a base -> similar content
                similar_group = ""
                base = None
                if ftype == "JPEG":
                    group = k // 2
                    if k % 2 == 0:
                        self._jpeg_bases[group] = self._new_jpeg_base(group)
                    base = self._jpeg_bases[group]
                    similar_group = f"jpeg-g{group}"  # both pair members tagged
                elif ftype == "PDF":
                    similar_group = f"pdf-g{k // 2}"  # shared template per pair
                name = f"{TYPE_PREFIX[ftype]}{idx:02d}.{ftype.lower()}"
                target = self.rng.randint(lo, hi)
                data = SYNTHESIZERS[ftype](self.rng, target, base)
                (ORIGINAL_FILES_DIR / name).write_bytes(data)
                self.files.append({"file_name": name, "file_type": ftype,
                                   "size_bytes": len(data), "similar_group": similar_group})
                idx += 1
        return self.files

    def _new_jpeg_base(self, group: int):
        rng = random.Random(self.seed * 1000 + group)
        w, h = 220, 220
        from PIL import Image
        img = Image.new("RGB", (w, h))
        px = img.load()
        r0, g0, b0 = rng.randrange(256), rng.randrange(256), rng.randrange(256)
        for y in range(h):
            for x in range(w):
                px[x, y] = ((r0 + x) % 256, (g0 + y) % 256, (b0 + x * y // 256) % 256)
        return img

    # -------------------------------------------------------------- fragments
    def build_fragments(self) -> None:
        # stratified train/holdout split by file type (doc §27 generalisation)
        by_type: Dict[str, List[Dict]] = {}
        for f in self.files:
            by_type.setdefault(f["file_type"], []).append(f)
        for ftype, flist in by_type.items():
            n_hold = max(1, round(len(flist) * HOLDOUT_FILE_FRACTION))
            for i, f in enumerate(flist):
                f["split"] = "holdout" if i < n_hold else "train"

        blob_dir = FRAGMENTS_DIR / "blobs"
        blob_dir.mkdir(parents=True, exist_ok=True)
        fid = 0
        for f in self.files:
            data = (ORIGINAL_FILES_DIR / str(f["file_name"])).read_bytes()
            n = (len(data) + FRAGMENT_SIZE - 1) // FRAGMENT_SIZE
            f["n_fragments"] = n
            for pos in range(n):
                fid += 1
                chunk = data[pos * FRAGMENT_SIZE:(pos + 1) * FRAGMENT_SIZE]
                if len(chunk) < FRAGMENT_SIZE:
                    chunk = chunk  # final partial fragment keeps true size
                fr_id = f"F{fid:05d}"
                self.blobs[fr_id] = chunk
                (blob_dir / f"{fr_id}.bin").write_bytes(chunk)
                self.fragments.append({
                    "fragment_id": fr_id, "source_file": f["file_name"],
                    "file_type": f["file_type"], "original_position": pos,
                    "fragment_size": len(chunk), "offset_in_source": pos * FRAGMENT_SIZE,
                    "split": f["split"], "similar_group": f["similar_group"],
                })

    # -------------------------------------------------------------- output
    def write_ground_truth(self) -> None:
        with open(GROUND_TRUTH_DIR / "files.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["file_name", "file_type", "size_bytes",
                                               "n_fragments", "split", "similar_group"])
            w.writeheader()
            for f in self.files:
                w.writerow({k: f.get(k, "") for k in w.fieldnames})
        with open(GROUND_TRUTH_DIR / "fragments.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["fragment_id", "source_file", "file_type",
                                               "original_position", "fragment_size",
                                               "offset_in_source", "split", "similar_group"])
            w.writeheader()
            w.writerows(self.fragments)


# --------------------------------------------------------------------------
#  Scenario image writer (doc §7)
# --------------------------------------------------------------------------

class ScenarioImageWriter:
    """Writes FSIM disk images with noise, decoys, corruption and gaps."""

    def __init__(self, corpus: CorpusBuilder, seed: int = SEED):
        self.corpus = corpus
        self.rng = random.Random(seed + 777)

    def _noise_block(self) -> bytes:
        return self.rng.randbytes(DISK_BLOCK_SIZE)

    def build_image(self, scenario_id: str, file_names: List[str], layout: str = "contiguous",
                    corruption_pct: float = 0.0, missing_pct: float = 0.0,
                    decoys: bool = False, description: str = "") -> Dict[str, object]:
        rng = self.rng
        # per-file fragment plans
        plans: Dict[str, List[Dict[str, object]]] = {}
        missing_rows: List[Dict[str, object]] = []
        for name in file_names:
            frags = [m for m in self.corpus.fragments if m["source_file"] == name]
            frags.sort(key=lambda m: m["original_position"])
            keep, missing = [], []
            for m in frags:
                is_edge = m["original_position"] in (0, len(frags) - 1)
                if missing_pct > 0 and not is_edge and rng.random() < missing_pct:
                    missing.append(dict(m, status="missing", physical_offset=-1, cluster=-1,
                                        allocation_status="unallocated", corruption_pct=0.0))
                else:
                    keep.append(m)
            plans[name] = keep
            missing_rows.extend(missing)

        # placement: sequential block cursor with interleaving
        header_blocks = SUPERBLOCK_BLOCKS + 64
        cursor = header_blocks
        placements: List[Dict[str, object]] = []          # ordered metadata rows
        interleave = FRAGMENTATION_INTERLEAVE.get(layout, 0)
        mixed = layout == "mixed"

        def emit_frag(m: Dict[str, object], status: str) -> None:
            nonlocal cursor
            placements.append(dict(m, status=status, physical_offset=cursor * DISK_BLOCK_SIZE,
                                   cluster=cursor, allocation_status="allocated",
                                   corruption_pct=corruption_pct))
            cursor += 1

        def emit_noise() -> None:
            nonlocal cursor
            placements.append({"fragment_id": "", "source_file": "", "file_type": "",
                               "original_position": -1, "fragment_size": DISK_BLOCK_SIZE,
                               "physical_offset": cursor * DISK_BLOCK_SIZE, "cluster": cursor,
                               "allocation_status": "unallocated", "corruption_pct": 0.0,
                               "status": "noise", "offset_in_source": -1, "split": "",
                               "similar_group": ""})
            cursor += 1

        order: List[Tuple[str, Dict[str, object]]] = []
        if mixed:
            # round-robin across files (scenario F) honouring per-file order
            iters = {n: 0 for n in file_names}
            remaining = True
            while remaining:
                remaining = False
                for name in file_names:
                    if iters[name] < len(plans[name]):
                        order.append((name, plans[name][iters[name]]))
                        iters[name] += 1
                        remaining = True
        else:
            for name in file_names:
                for m in plans[name]:
                    order.append((name, m))

        for i, (name, m) in enumerate(order):
            emit_frag(m, "present")
            if mixed and rng.random() < 0.35:
                emit_noise()
            elif interleave and rng.random() < 0.85:
                for _ in range(rng.randint(1, interleave)):
                    emit_noise()

        # decoy fragments (competing candidates, doc §32/H): real fragments of
        # same type from files NOT in this scenario, planted near similar files
        decoy_rows: List[Dict[str, object]] = []
        if decoys:
            candidates = [m for m in self.corpus.fragments
                          if m["source_file"] not in file_names and m["original_position"] < 4]
            rng.shuffle(candidates)
            insert_points = [p for p in range(len(placements))
                             if placements[p]["fragment_id"]][:60]
            for m in candidates[: min(20, len(candidates))]:
                at = rng.choice(insert_points) if insert_points else len(placements)
                decoy = dict(m, status="decoy", physical_offset=-1, cluster=-1,
                             allocation_status="allocated", corruption_pct=0.0)
                placements.insert(min(at, len(placements)), decoy)
                decoy_rows.append(decoy)
        placements.extend(missing_rows)  # gaps recorded with offset -1

        # assign physical offsets in final (possibly shuffled-insert) order
        final_cursor = header_blocks
        for row in placements:
            if row["status"] in ("present", "decoy"):
                row["physical_offset"] = final_cursor * DISK_BLOCK_SIZE
                row["cluster"] = final_cursor
            final_cursor += 1
        total_blocks = final_cursor + 32

        # render image
        image = bytearray(total_blocks * DISK_BLOCK_SIZE)
        bitmap = bytearray((total_blocks + 7) // 8)

        def mark_alloc(block_i: int) -> None:
            bitmap[block_i // 8] |= 1 << (7 - (block_i % 8))

        # superblock
        volume_id = f"FS{scenario_id}".encode()[:16].ljust(16, b"\x00")
        sb = struct.pack("<8sIIIII16s", FSIM_MAGIC, DISK_BLOCK_SIZE, total_blocks,
                         SUPERBLOCK_BLOCKS, 64, DISK_BLOCK_SIZE, volume_id)
        image[0:len(sb)] = sb
        for b in range(SUPERBLOCK_BLOCKS, SUPERBLOCK_BLOCKS + 64):
            mark_alloc(b)

        for row in placements:
            off, status = row["physical_offset"], row["status"]
            block_i = off // DISK_BLOCK_SIZE
            if status == "noise":
                image[off:off + DISK_BLOCK_SIZE] = self._noise_block() if rng.random() < 0.7 \
                    else bytes(DISK_BLOCK_SIZE)
                continue
            frag_id = str(row["fragment_id"])
            data = bytearray(self.corpus.blobs[frag_id])
            if status == "present" and corruption_pct > 0:
                n_corrupt = int(round(len(data) * corruption_pct))
                idxs = rng.sample(range(len(data)), n_corrupt)
                noise = rng.randbytes(n_corrupt)
                for j, ix in enumerate(idxs):
                    data[ix] = noise[j]
            image[off:off + len(data)] = data
            mark_alloc(block_i)

        # write allocation bitmap into its header blocks
        bmp_off = SUPERBLOCK_BLOCKS * DISK_BLOCK_SIZE
        image[bmp_off:bmp_off + len(bitmap)] = bitmap

        image_path = DISK_IMAGES_DIR / f"{scenario_id}.img"
        image_path.write_bytes(bytes(image))

        # ground truth for this scenario
        sdir = SCENARIOS_DIR / scenario_id
        sdir.mkdir(parents=True, exist_ok=True)
        fields = ["fragment_id", "source_file", "file_type", "original_position",
                  "fragment_size", "physical_offset", "cluster", "allocation_status",
                  "corruption_pct", "status", "offset_in_source", "split", "similar_group"]
        with open(sdir / "placements.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for row in placements:
                w.writerow({k: row.get(k, "") for k in fields})

        present = [r for r in placements if r["status"] == "present"]
        manifest = {
            "scenario_id": scenario_id, "description": description, "layout": layout,
            "corruption_pct": corruption_pct, "missing_pct": missing_pct, "decoys": decoys,
            "files": file_names, "image_path": str(image_path),
            "placements_path": str(sdir / "placements.csv"),
            "total_blocks": total_blocks, "image_bytes": len(image),
            "n_present": len(present), "n_missing": sum(1 for r in placements if r["status"] == "missing"),
            "n_decoys": len(decoy_rows), "n_noise": sum(1 for r in placements if r["status"] == "noise"),
            "n_fragments": len(present) + len(decoy_rows),
        }
        save_json(sdir / "manifest.json", manifest)
        logger.info("scenario %-16s blocks=%-6d present=%-4d missing=%-3d decoys=%-3d noise=%d",
                    scenario_id, total_blocks, manifest["n_present"], manifest["n_missing"],
                    manifest["n_decoys"], manifest["n_noise"])
        return manifest


def _split_by_type(files: List[Dict[str, object]]) -> Tuple[List[str], List[str]]:
    train = [f["file_name"] for f in files if f.get("split") == "train"]
    hold = [f["file_name"] for f in files if f.get("split") == "holdout"]
    return train, hold


def generate_all(seed: int = SEED) -> Dict[str, object]:
    """Generate the complete controlled corpus and all scenario images."""
    set_seeds(seed)
    with timer("corpus generation", logger):
        builder = CorpusBuilder(seed)
        builder.generate_files()
        builder.build_fragments()
        builder.write_ground_truth()
        logger.info("files=%d fragments=%d", len(builder.files), len(builder.fragments))

        writer = ScenarioImageWriter(builder, seed)
        train, hold = _split_by_type(builder.files)
        similar_train = [f["file_name"] for f in builder.files
                         if f.get("similar_group") and f.get("split") == "train"]

        manifests: Dict[str, object] = {}
        manifests["S_A"] = writer.build_image("S_A", train, "contiguous", 0, 0, False,
                                              "Scenario A - contiguous layout (doc §7)")
        manifests["S_B"] = writer.build_image("S_B", train, "low", 0, 0, False,
                                              "Scenario B - low fragmentation")
        manifests["S_C"] = writer.build_image("S_C", train, "high", 0, 0, False,
                                              "Scenario C - high fragmentation")
        for pct in CORRUPTION_LEVELS:
            sid = f"S_D{int(pct * 100):02d}"
            manifests[sid] = writer.build_image(sid, train, "contiguous", pct, 0, False,
                                                f"Scenario D - {pct:.0%} byte corruption")
        for pct in MISSING_LEVELS:
            sid = f"S_E{int(pct * 100):02d}"
            manifests[sid] = writer.build_image(sid, train, "contiguous", 0, pct, False,
                                                f"Scenario E - {pct:.0%} missing fragments")
        manifests["S_F"] = writer.build_image("S_F", train, "mixed", 0, 0, False,
                                              "Scenario F - mixed-file interleaving")
        manifests["S_G"] = writer.build_image("S_G", similar_train, "medium", 0, 0, False,
                                              "Scenario G - similar-content files")
        manifests["S_H"] = writer.build_image("S_H", train, "severe", 0.10, 0.10, True,
                                              "Scenario H - combined adversarial")

        # evaluation / generalisation images (unseen files, doc §27)
        manifests["EVAL_A"] = writer.build_image("EVAL_A", hold, "contiguous", 0, 0, False,
                                                 "Holdout - contiguous")
        manifests["EVAL_C"] = writer.build_image("EVAL_C", hold, "high", 0, 0, False,
                                                 "Holdout - high fragmentation")
        manifests["EVAL_H"] = writer.build_image("EVAL_H", hold, "severe", 0.10, 0.10, True,
                                                 "Holdout - adversarial")
        # sweeps (doc §29-33)
        for lvl in ("low", "medium", "severe"):
            sid = f"SWEEP_FRAG_{lvl.upper()}"
            manifests[sid] = writer.build_image(sid, hold, lvl, 0, 0, False,
                                                f"Fragmentation sweep - {lvl}")
        for pct in CORRUPTION_LEVELS:
            sid = f"SWEEP_CORR_{int(pct * 100):02d}"
            manifests[sid] = writer.build_image(sid, hold, "contiguous", pct, 0, False,
                                                f"Corruption sweep - {pct:.0%}")
        for pct in MISSING_LEVELS:
            sid = f"SWEEP_MISS_{int(pct * 100):02d}"
            manifests[sid] = writer.build_image(sid, hold, "contiguous", 0, pct, False,
                                                f"Missing sweep - {pct:.0%}")
        manifests["SWEEP_DECOYS"] = writer.build_image("SWEEP_DECOYS", hold, "contiguous", 0, 0, True,
                                                       "Competing fragments sweep")

        summary = {"n_files": len(builder.files), "n_fragments": len(builder.fragments),
                   "n_train_files": len(train), "n_holdout_files": len(hold),
                   "fragment_size": FRAGMENT_SIZE, "scenarios": {k: v for k, v in manifests.items()}}
        save_json(SCENARIOS_DIR / "corpus_summary.json", summary)
        print(f"[forensight] corpus ready: {summary['n_files']} files, "
              f"{summary['n_fragments']} fragments, {len(manifests)} scenario images")
        return summary


if __name__ == "__main__":
    generate_all()
