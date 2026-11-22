# External Digital Forensic Dataset Registry & Suitability Assessment

**ForenSight Research Framework — Phase 2**  
**Document Revision:** `v2.0.0-phase2`  
**Standard:** NIST / DFRWS Forensic Data Evaluation Guidelines  

---

## 1. Executive Summary & Assessment Criteria

In forensic file carving research, benchmark datasets must be rigorously assessed for ground-truth availability, licensing, and structural integrity. A corpus containing files is **not** inherently suitable for file reconstruction evaluation unless:
1. Ground-truth fragment ordering and byte-level file boundaries are cryptographically documented.
2. The licensing explicitly permits academic research, automated benchmarking, and derivative fragment generation.
3. The raw data is free of non-consensual PII, classified material, or proprietary copyright encumbrances.
4. Physical disk fragmentation can be verified against known disk images or synthesized under controlled conditions without breaking file format integrity.

This registry assesses the four primary publicly available forensic corpora, establishing an integration specification for the ForenSight external evaluation adapter.

---

## 2. Comprehensive Corpus Registry

### 2.1. Govdocs1 (Digital Corpora / Simson Garfinkel et al.)

- **Repository & Availability:** [digitalcorpora.org/corpora/files](https://digitalcorpora.org/corpora/files) (Hosted via Harvard / AWS Open Data).
- **License & Permitted Use:** Creative Commons CC0 1.0 Universal (Public Domain Dedication). Unrestricted commercial and academic use.
- **Corpus Scale & Contents:** ~1,000,000 documents collected from `.gov` web crawls (1,000 numbered threads of 1,000 files each).
- **Supported Formats:** PDF, DOC, DOCX, PPT, PPTX, XLS, XLSX, HTML, TXT, XML, JPG, PNG, GIF, EPS, SWF.
- **Ground-Truth Availability:** Complete SHA-1, MD5, and file size manifests available via `govdocs1-manifest.csv`.
- **Pre-fragmented Disk Images:** No. Govdocs1 provides individual unfragmented files.
- **Controlled Fragmentation Suitability:** **Excellent (High)**. Unfragmented original files can be fragmented into arbitrary sector sizes (512B, 4096B) and interleaved across simulated raw disk images (`.raw`, `.dd`, `.E01`).
- **Exact Reconstruction Measurement:** Fully feasible. Reconstructed byte streams are hashed (SHA-256) and verified against original source manifests.
- **Adoption Recommendation:** **Approved for External Benchmark.** Recommended sampling: Govdocs1 Thread 000 (1,000 files) filtered to target formats (PDF, DOCX, XLSX, JPEG, PNG).

---

### 2.2. DFRWS Forensic Challenge Datasets (2006, 2007, 2008)

- **Repository & Availability:** [dfrws.org/dfrws-forensic-challenges](https://dfrws.org/dfrws-forensic-challenges) (DFRWS 2006: File Carving Challenge; DFRWS 2007: Memory Forensics; DFRWS 2008: P2P Carving).
- **License & Permitted Use:** Open Academic Research License. Freely usable for forensic algorithm comparison and citation.
- **Corpus Scale & Contents:**
  - *DFRWS 2006*: 50 MB raw FAT disk image (`dfrws-2006-challenge.raw`) designed specifically to test fragmented carving.
  - *DFRWS 2007*: Memory dumps containing fragmented payload structures.
- **Supported Formats:** JPEG, ZIP, PDF, DOC, GIF, HTML, PNG.
- **Ground-Truth Availability:** Full ground-truth sector map provided by challenge authors (identifies exact cluster offsets, fragmented sectors, missing clusters, and target file hashes).
- **Pre-fragmented Disk Images:** **Yes**. Specifically designed with realistic FAT16/FAT32 fragmentation, multi-fragment interleaving, and missing headers.
- **Controlled Fragmentation Suitability:** Fixed challenge images; secondary synthetic fragmentation can also be introduced using raw extracted ground-truth files.
- **Exact Reconstruction Measurement:** Fully feasible against challenge answer keys (`answer_key_dfrws2006.txt`).
- **Adoption Recommendation:** **Approved as Standard Real-World Benchmark.** Essential for benchmarking against legacy carvers (Scalpel, Foremost, PhotoRec).

---

### 2.3. NIST Computer Forensic Tool Testing (CFTT) / CFReDS

- **Repository & Availability:** [cfreds.nist.gov](https://cfreds.nist.gov/) (NIST Computer Forensic Reference Data Sets).
- **License & Permitted Use:** US Government Public Domain (17 U.S.C. § 105). Unrestricted redistribution and modification.
- **Corpus Scale & Contents:**
  - *CFTT File Carving Test Images*: Disk images containing contiguous, fragmented, and interleaved test files.
  - *CFTT Deleted File Recovery Images*: Ext2, Ext3, NTFS, and FAT filesystem images.
- **Supported Formats:** JPEG, TIFF, BMP, PDF, DOC, XLS, ZIP.
- **Ground-Truth Availability:** Detailed NIST Test Case Documentation (`CFTT-Carving-Test-Plan.pdf`), sector-level allocation tables, MD5 manifests.
- **Pre-fragmented Disk Images:** **Yes**. Multi-sector fragmentation, slack space carving, and header-less fragment tests.
- **Controlled Fragmentation Suitability:** High. Fixed test images test specific standard compliance criteria (e.g., CARV-01 through CARV-12).
- **Exact Reconstruction Measurement:** Verified through official NIST test evaluation criteria.
- **Adoption Recommendation:** **Approved for Regulatory / Tool Testing Benchmark.**

---

### 2.4. SANS Forensic Contest Corpora / Lone Wolf Scenario

- **Repository & Availability:** Network and disk challenges hosted by SANS Institute / independent digital forensics challenges.
- **License & Permitted Use:** Educational / Personal Use Only. Redistribution of full disk images is restricted under terms of service.
- **Corpus Scale & Contents:** Large enterprise disk images (10 GB – 100 GB) containing OS installations, malware, and fragmented evidence.
- **Ground-Truth Availability:** Narrative challenge solutions; sector-level fragment manifests are generally incomplete or unverified.
- **Pre-fragmented Disk Images:** Yes, realistic NTFS/Ext4 disk images.
- **Controlled Fragmentation Suitability:** Poor. Huge operational overhead with ambiguous file boundary ground truths.
- **Exact Reconstruction Measurement:** Difficult due to incomplete ground truth on ambient background files.
- **Adoption Recommendation:** **Rejected for Automated Benchmarking.** Not suitable for automated, reproducible fragment metric evaluation.

---

## 3. Comparative Dataset Matrix

| Dataset | License | Scale | Pre-Fragmented? | Ground-Truth Verification | Exact Hash Match Feasible? | Suitability Tier |
|---|---|---|---|---|---|---|
| **Govdocs1** | CC0 (Public Domain) | 1,000,000 files | No (Raw Files) | Cryptographic Manifests | Yes | **Tier 1 (Synthetic Carving)** |
| **DFRWS 2006** | Open Academic | 50 MB Disk Image | Yes (FAT32) | Verified Sector Maps | Yes | **Tier 1 (Real Carving)** |
| **NIST CFTT** | US Gov Public Domain | ~20 Disk Images | Yes (Multiple FS) | Detailed Sector Tables | Yes | **Tier 1 (Standard Verification)** |
| **SANS Lone Wolf** | Restricted / Educational | 40 GB Image | Yes (Complex OS) | Narrative Only | Partial | **Rejected** |

---

## 4. External Dataset Adapter Architecture

To maintain the **Absolute Preservation Policy** and ensure that external evidence does not contaminate synthetic benchmarks, ForenSight implements an adapter pattern:

```
[ External Repository ] (e.g. Govdocs1 / DFRWS 2006)
          |
          v
[ ExternalDatasetAdapter ]
    +-- verify_license()
    +-- download_or_load_local()
    +-- parse_manifest() -> SectorGroundTruth
    +-- extract_observable_blocks() -> ObservableFragment
          |
          v
[ ForenSight Pipeline (v2.0.0-phase2) ]
    +-- LeakageFreeFeatureEngine
    +-- EvaluationEngine (Evaluates exact reconstruction & partial metrics)
```

### Safety Rules:
1. External evidence records must be validated using `validate_observable_record()`.
2. External evaluation metrics must be saved to separate JSON files (`results/external_eval_*.json`) and never merged with synthetic benchmark results.
3. No proprietary, unverified, or restricted corpora may be incorporated.
