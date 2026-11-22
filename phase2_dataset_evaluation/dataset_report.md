# PHASE 2 — EXPANDED FORENSIC DATASET REPORT

**ForenSight Research Framework — Phase 2**  
**Document Revision:** `v2.0.0-phase2`  
**Date:** 2026-11-22  
**Dataset Version:** `2.0.0-phase2`  
**Storage Path:** `phase2_dataset_evaluation/dataset_v2/`  

---

## 1. Executive Summary

This report documents the implementation and validation of the expanded digital forensic fragment dataset (`dataset_v2`) for the ForenSight research framework.

In accordance with the **Absolute Preservation Policy**, the original 50-file synthetic dataset, all existing disk images, and scenario manifests remain 100% immutable and untouched in `dataset/`. The new expanded corpus is hosted in `phase2_dataset_evaluation/dataset_v2/`.

### Key Metrics of Dataset v2:
- **Total Original Files**: 72 structurally diverse files across 8 formats.
- **Total Carved Fragments**: 1,464 fragments (at 512-byte block granularity).
- **Total Payload Size**: ~750 KB of diverse binary and structural payloads.
- **Formats Represented**: JPEG (10), PNG (10), PDF (10), ZIP (10), DOCX (8), XLSX (8), MP3 (8), MP4 (8).
- **Physical Layout Decoupling**: Complete independence between ground-truth logical sequence and simulated physical disk sectors (LBA).
- **Cryptographic Ground Truth**: 100% of files and fragments hashed with SHA-256 and cataloged in `dataset_manifest.json` and `ground_truth/fragments_v2.csv`.

---

## 2. Format-by-Format Structural Diversity

Rather than duplicating files by renaming identical payloads, Dataset v2 introduces genuine structural, encoding, and metadata diversity across all 8 file formats:

### 2.1. JPEG (10 Files)
- **Baseline vs. Progressive**: 6 baseline DCT encoded files vs. 4 progressive spectral selection files (`progressive=True`).
- **Color Spaces**: 8 24-bit RGB truecolor files, 2 8-bit Grayscale (`L`) files.
- **Compression Factors**: Ranging from severe compression ($Q=50$) with high blocking artifacts, to medium ($Q=65, 75$), to fine quality ($Q=85, 90, 95$).
- **Metadata Markers**: Embedded EXIF/COM comment marker chunks (`ForenSight_Evidence_Corpus_v2`).
- **Dimensions**: Small icons (96x96), square graphics (200x200, 240x240), wide landscape banners (300x150, 400x250).

### 2.2. PNG (10 Files)
- **Color Types**: 24-bit RGB (truecolor), 32-bit RGBA (alpha transparency), 8-bit Grayscale (`L`), 8-bit Palette-indexed (`P`) with 256-color adaptive quantization.
- **Ancillary Chunks**: Chunks including `tEXt` (Author, Description), `pHYs` (physical pixel dimensions), and `gAMA` (display gamma 2.2).
- **Compression & Filtering**: Standard Deflate (level 6) vs. High compression (level 9) with adaptive sub/up/average/paeth row filters.
- **Interlacing**: Adam7 interlaced PNG streams.

### 2.3. PDF (10 Files — Generated via ReportLab)
- **Document Versions**: PDF 1.4, 1.5, and 1.7.
- **Structural Layouts**: Multi-page flowables (1 to 3 pages), structured paragraphs, tabular grids (`Table` flowables with styled borders and headers).
- **Graphics Content**: Direct vector graphics canvas operations (rectangles, lines, circles, coordinate transforms).
- **Stream Compression**: FlateDecode compressed object streams and traditional uncompressed cross-reference (`xref`) tables.

### 2.4. ZIP (10 Files)
- **Compression Methods**: `ZIP_DEFLATED` (standard LZ77/Huffman) vs. `ZIP_STORED` (uncompressed raw stream).
- **Archive Structures**: Single large payloads, archives with 20 small files, nested multi-tier directories (`docs/reports/2026/`).
- **Member Types**: Mixed binary streams (`.bin`), structured JSON (`.json`), plain text (`.txt`), and CSV logs (`.csv`).

### 2.5. DOCX (8 Files — Generated via python-docx)
- **OOXML Package Structures**: Distinct `word/document.xml`, `[Content_Types].xml`, `_rels/.rels`.
- **Document Content**: Multi-column tabular inventories, styled headings (Title, Heading 1, Heading 2), bold/italic run variations.
- **Metadata**: Embedded core properties (`core.xml`) with custom Author, Classification, and Comments tags.

### 2.6. XLSX (8 Files — Generated via openpyxl)
- **Worksheet Layouts**: Single-sheet workbooks vs. multi-sheet workbooks (`Summary`, `Q1`, `Q2`, `Secondary Metrics`).
- **Formulas**: Cell formula references (`=SUM(...)`, `=AVERAGE(...)`, `=A2*2`).
- **Shared Strings**: Reusable string tables (`xl/sharedStrings.xml`) creating cross-fragment dependencies.
- **Cell Styling**: Colored background fills, borders, font weights, and percentage formats.

### 2.7. MP3 (8 Files)
- **Bitrates**: Variable CBR profiles (64 kbps mono speech, 128 kbps, 192 kbps, 256 kbps, 320 kbps high fidelity).
- **Sampling Rates**: 44.1 kHz and 48.0 kHz sampling frequency frame sync headers (`0xFFFB`, `0xFFFA`).
- **Metadata Containers**: ID3v2.3 tags with `TIT2` title frames, raw headerless MP3 streams, and trailing 128-byte ID3v1 `TAG` blocks.

### 2.8. MP4 (8 Files)
- **Brand Containers**: `ftyp` box variations (`isom` vs `mp42`).
- **Box Hierarchy**: Standard `ftyp -> moov -> mdat` order vs. streaming-optimized `ftyp -> mdat -> moov` (fragmented atoms).
- **Track Layouts**: Video-only track atoms (`trak: vide`), audio-only track atoms (`trak: soun`), and dual-stream multimedia containers.
- **Offset Tables**: 32-bit chunk offset atoms (`stco`) and 64-bit large file offset atoms (`co64`).

---

## 3. Physical Disk Layout Architecture

The `PhysicalDiskLayoutGenerator` implements a realistic storage media layout:

```
[ Sector 0: VBR / Superblock (FSIM_V20, BlockSize, TotalSectors, DataSectors) ]
[ Sector 1-7: Allocation Bitmaps / System Metadata ]
[ Sector 8 - N: Data Area & Slack Space Pool ]
    |-- File A Run (Cluster 0..3)
    |-- Foreign Gap / Slack Filler (Unallocated)
    |-- File B Run (Cluster 0..3)
    |-- File A Run (Cluster 4..7)   <-- Interleaved Run
    |-- Decoy Fragment (Injected in Unallocated Slack)
```

### Decoupling Guarantee:
- The **observable sector stream** presents only physical sector LBA, raw bytes, and allocation status (allocated/unallocated).
- Ground truth (logical sequence, source file identity, corruption flags, and decoy labels) is stored strictly in `ground_truth_manifest.json` and cannot be observed during inference.

---

## 4. Scenario Matrix Coverage

The framework implements 11 standardized, reproducible experimental scenarios:

| Scenario ID | Name | Interleave ($I$) | Run Length | Corruption | Missing | Decoys |
|---|---|---|---|---|---|---|
| `SCEN_A_CONTIG_CLEAN` | Baseline Contiguous | 0 | 64 | 0% | 0% | 0 |
| `SCEN_B_LOW_FRAG_2F` | Low Fragmentation | 1 | 8 | 0% | 0% | 0 |
| `SCEN_C_MED_FRAG_5F` | Medium Fragmentation | 3 | 4 | 0% | 0% | 0 |
| `SCEN_D_HIGH_FRAG_10F` | High Fragmentation | 6 | 2 | 0% | 0% | 0 |
| `SCEN_E_EXTREME_FRAG_15F` | Extreme Fragmentation | 12 | 1 | 0% | 0% | 0 |
| `SCEN_F_CORRUPT_C05` | Payload Corruption 5% | 2 | 4 | 5% | 0% | 0 |
| `SCEN_G_CORRUPT_C20` | Payload Corruption 20% | 2 | 4 | 20% | 0% | 0 |
| `SCEN_H_MISSING_M05` | Missing Fragments 5% | 2 | 4 | 0% | 5% | 0 |
| `SCEN_I_MISSING_M20` | Missing Fragments 20% | 2 | 4 | 0% | 20% | 0 |
| `SCEN_J_DECOY_D15` | Decoys Injection | 2 | 4 | 0% | 0% | 15 |
| `SCEN_K_ADVERSARIAL_STORM` | Adversarial Stress Test | 5 | 2 | 10% | 10% | 20 |

---

## 5. Artifact Preservation Verification

- Original files in `dataset/original_files/`: **Preserved (50 files)**
- Original fragments in `dataset/fragments/`: **Preserved (3,538 fragments)**
- Original ground truth in `dataset/ground_truth/`: **Preserved**
- New expanded files in `phase2_dataset_evaluation/dataset_v2/`: **Verified (72 files, 1,464 fragments)**
- Git tracked file state: **0 tracked files modified**.
