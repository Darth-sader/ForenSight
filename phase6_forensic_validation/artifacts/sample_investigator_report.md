# FORENSIC FILE RECONSTRUCTION AUDIT REPORT
**Case / Evidence Identifier:** DISK_IMAGE_001
**Reconstruction ID:** `recon_archive27.zip` | **Software Version:** `v2.0.0-phase6`
**Generated (UTC):** 2026-11-23T15:25:04Z | **Acquisition Source:** PHYSICAL_RAW

---

## 1. EVIDENTIARY VERDICT & SUMMARY
- **Disposition Outcome:** **`ACCEPTED`**
- **Structural Validation Tier:** `L5_L5_GROUND_TRUTH`
- **Claimed Format:** `ZIP`
- **Reconstructed Size:** `6,656 bytes` (13 fragments)
- **SHA-256 Digest:** `f79e3556911eb8058cdeda477c62a0e49a8597328403d0fdadb9a8fd7c30fe5f`
- **MD5 Digest:** `075f63ac0fd2782655f96d3b8a3200d1`
- **Decision Rationale:** Meets full evidence criteria: structural parsing passed, zero unresolved gaps, low uncertainty.
- **Court / Investigative Admissibility:** Qualified for investigative use

---

## 2. COMPONENT FRAGMENT CHAIN OF CUSTODY
| Position | Fragment ID | Physical Offset | Length (Bytes) | Allocation Status | SHA-256 Hash (First 16 chars) |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 00 | `archive27.zip_frag_0000` | `0` | `512` | `unallocated` | `fcd333bf5acdf11e...` |
| 01 | `archive27.zip_frag_0001` | `512` | `512` | `unallocated` | `c0872aaea45980d1...` |
| 02 | `archive27.zip_frag_0002` | `1,024` | `512` | `unallocated` | `c2e2d175aa24caf5...` |
| 03 | `archive27.zip_frag_0003` | `1,536` | `512` | `unallocated` | `9c28faafa60c2d0c...` |
| 04 | `archive27.zip_frag_0004` | `2,048` | `512` | `unallocated` | `832b428b74262a5f...` |
| 05 | `archive27.zip_frag_0005` | `2,560` | `512` | `unallocated` | `475b3fb79872ca00...` |
| 06 | `archive27.zip_frag_0006` | `3,072` | `512` | `unallocated` | `14004902822f52b8...` |
| 07 | `archive27.zip_frag_0007` | `3,584` | `512` | `unallocated` | `97472db076dd0ab7...` |
| 08 | `archive27.zip_frag_0008` | `4,096` | `512` | `unallocated` | `5b11b6a59be55115...` |
| 09 | `archive27.zip_frag_0009` | `4,608` | `512` | `unallocated` | `e8bec415c4aac59f...` |
| 10 | `archive27.zip_frag_0010` | `5,120` | `512` | `unallocated` | `fc9fb23cc7f42842...` |
| 11 | `archive27.zip_frag_0011` | `5,632` | `512` | `unallocated` | `09892be040890b50...` |
| 12 | `archive27.zip_frag_0012` | `6,144` | `512` | `unallocated` | `e18e1a8c718328e9...` |

---

## 3. UNCERTAINTY & TRANSITION CONFIDENCE BREAKDOWN
> [!NOTE]
> All confidence scores distinguish raw model probabilities from calibrated likelihoods and graph scores. > Probabilities are never presented as proofs of factual authenticity.

| Transition | Type | Raw Probability | Calibrated Probability | Graph Score | Optimization Decision |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `archive27.zip_frag_0000` -> `archive27.zip_frag_0001` | `DIRECT_SUCCESSOR` | `0.9000` | `0.8800` | `0.8800` | `SELECTED` |
| `archive27.zip_frag_0001` -> `archive27.zip_frag_0002` | `DIRECT_SUCCESSOR` | `0.9000` | `0.8800` | `0.8800` | `SELECTED` |
| `archive27.zip_frag_0002` -> `archive27.zip_frag_0003` | `DIRECT_SUCCESSOR` | `0.9000` | `0.8800` | `0.8800` | `SELECTED` |
| `archive27.zip_frag_0003` -> `archive27.zip_frag_0004` | `DIRECT_SUCCESSOR` | `0.9000` | `0.8800` | `0.8800` | `SELECTED` |
| `archive27.zip_frag_0004` -> `archive27.zip_frag_0005` | `DIRECT_SUCCESSOR` | `0.9000` | `0.8800` | `0.8800` | `SELECTED` |
| `archive27.zip_frag_0005` -> `archive27.zip_frag_0006` | `DIRECT_SUCCESSOR` | `0.9000` | `0.8800` | `0.8800` | `SELECTED` |
| `archive27.zip_frag_0006` -> `archive27.zip_frag_0007` | `DIRECT_SUCCESSOR` | `0.9000` | `0.8800` | `0.8800` | `SELECTED` |
| `archive27.zip_frag_0007` -> `archive27.zip_frag_0008` | `DIRECT_SUCCESSOR` | `0.9000` | `0.8800` | `0.8800` | `SELECTED` |
| `archive27.zip_frag_0008` -> `archive27.zip_frag_0009` | `DIRECT_SUCCESSOR` | `0.9000` | `0.8800` | `0.8800` | `SELECTED` |
| `archive27.zip_frag_0009` -> `archive27.zip_frag_0010` | `DIRECT_SUCCESSOR` | `0.9000` | `0.8800` | `0.8800` | `SELECTED` |
| `archive27.zip_frag_0010` -> `archive27.zip_frag_0011` | `DIRECT_SUCCESSOR` | `0.9000` | `0.8800` | `0.8800` | `SELECTED` |
| `archive27.zip_frag_0011` -> `archive27.zip_frag_0012` | `DIRECT_SUCCESSOR` | `0.9000` | `0.8800` | `0.8800` | `SELECTED` |

### Multidimensional Uncertainty Signals
- **Predictive Entropy:** `0.4690` (bits)
- **Model Disagreement (Variance):** `0.0000`
- **Alternative Path Margin:** `1.2000`
- **Graph Topological Ambiguity:** `0.0500`
- **Missing Gap Ratio:** `0.00`
- **Composite Uncertainty Score:** **`0.2384`** (Scale: 0.0=Certain to 1.0=Uncertain)

---

## 4. FORMAT STRUCTURAL & PARSER VALIDATION AUDIT
- **Format Validator Executed:** `ZIP`
- **Highest Verified Tier:** `L5_L5_GROUND_TRUTH`
- **Structural Syntax Passed (L2):** `True`
- **Decompression / Renderable (L3):** `True`
- **Reference Ground Truth Match (L5):** `True`

### Validator Disclosures and Warnings:
- *No structural errors or warnings reported by independent parser.*

### Parsed Technical Metadata:
```json
{
  "entries": [
    "notes_0.txt",
    "notes_1.txt",
    "notes_2.txt",
    "notes_3.txt",
    "data_4.csv",
    "blob.bin"
  ],
  "entry_count": 6,
  "total_uncompressed_bytes": 134167
}
```

---

## 5. UNRESOLVED GAPS & MISSING EVIDENCE DISCLOSURE
- *Zero unresolved gaps detected. File is contiguous or completely bridged.*

---

## 7. CRYPTOGRAPHIC PROVENANCE & AUDIT REPRODUCIBILITY
- **Provenance Record Hash:** `06f17f4ab74f3c6ee5af063f22f4485f56b1f54308282c2304732137e1a11dbd`
- **Reproducibility Statement:** *This reconstruction can be bit-for-bit reproduced by executing the ForenSight optimization pipeline against the specified input fragment hashes under software version v2.0.0-phase6.*

> [!IMPORTANT]
> **MANDATORY SCIENTIFIC & FORENSIC DISCLAIMER:**
> Machine-learning predictions and probability estimates generated by ForenSight reflect statistical likelihoods > derived from trained models and observable evidence. They do **NOT** constitute definitive proof of factual > authenticity, file origin, or authorship. All reconstructed digital evidence must be independently corroborated > by qualified digital forensics examiners prior to presentation in legal proceedings.