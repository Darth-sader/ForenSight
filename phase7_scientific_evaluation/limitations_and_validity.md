# FORENSIGHT SCIENTIFIC EVALUATION
## THREATS TO VALIDITY & METHODOLOGICAL LIMITATIONS

**Target Manuscript:** *Forensic Science International: Digital Investigation* (FSI:DI)  
**Protocol Version:** `v2.0.0-phase7`  
**Date:** November 23, 2026  
**Document Status:** Final Audit Component  

---

## 1. INTERNAL VALIDITY

Internal validity assesses whether the observed experimental outcomes are attributable solely to the proposed algorithmic interventions rather than confounding factors, leakage, or instrumentation bugs.

### 1.1 Data Leakage Elimination
- **Mitigation:** In Phase 1, ForenSight completely overhauled the fragment data schema, establishing a hard structural partition between `GroundTruthFragment` and `ObservableFragment`. The inference pipeline (`FeatureEngine`, `CandidateGenerator`, `Models`, `Solvers`) receives strictly observable physical offsets and raw bytes.
- **Verification:** Unit tests (`test_schema_enforcement`, `test_provenance_isolation`) programmatically assert that any inference module attempting to read ground-truth fields (`logical_offset`, `logical_sequence_index`, `source_file_id`) raises a `GroundTruthLeakageError`.

### 1.2 Exchangeability Violations in Conformal Prediction
- **Identified Threat:** Standard split conformal prediction mathematically assumes that calibration and test samples are independently and identically distributed (i.i.d.) and exchangeable. In digital forensics, fragments originate from the same underlying files, and fragment pairs share common nodes in candidate graphs.
- **Empirical Impact:** As demonstrated in Phase 6, empirical nonconformity coverage on the held-out test partition dropped from the nominal $1 - \alpha = 0.95$ to $89.14\%$.
- **Scientific Honesty:** ForenSight explicitly documents that naive pairwise conformal prediction is invalid without cluster-adjusted or margin-deflated conformal bounds. Claim C04 is formally marked as **Unsupported** in the publication claim matrix.

### 1.3 Solver Timeouts and Optimality Gaps
- **Constraint:** The HiGHS MILP solver is bounded by a 60-second execution time limit per sub-problem. On extremely dense clusters without clear format guards, the solver may terminate at a feasible integer solution rather than proving global optimality.
- **Mitigation:** Candidate cluster decomposition ensures that typical sub-problems contain $<40$ fragments, where HiGHS routinely proves global optimality ($\text{gap} = 0.000$) in under 0.25 seconds.

---

## 2. EXTERNAL VALIDITY

External validity assesses whether the empirical findings generalize to operational forensic casework, commercial storage media, and unstandardized digital evidence.

### 2.1 Synthetic vs Physical Casework Images
- **Threat:** Evaluation was performed on controlled disk layouts (`dataset_v2` and synthetic scenario generators) rather than raw forensic physical clones seized from criminal casework.
- **Generalization Boundary:** Real storage devices feature file system metadata (NTFS Master File Table, ext4 inodes, APFS catalog trees), wear leveling, TRIM garbage collection, and disk slack spaces that may provide supplementary structural context not present in raw unallocated carving pools.
- **casework Caveat:** ForenSight is designed as an unallocated chunk carver when metadata is destroyed; evaluations on raw streams establish the lower bound of performance.

### 2.2 File Size & Fragment Count Distributions
- **Observation:** The test partition contains files spanning 512 bytes to 1.2 MB ($N = 955$ fragments).
- **Scalability Limitation:** Extremely large multimedia files (e.g., 4 GB MP4 videos comprising 8,000,000 sectors) cannot be modeled as a single monolithic MILP. For massive files, ForenSight must operate in streaming cluster windows.

### 2.3 Format Scope
- **Coverage:** Evaluated on 8 ubiquitous formats: JPEG, PNG, PDF, ZIP, DOCX, XLSX, MP3, and MP4.
- **Casework Reality:** Casework frequently encounters proprietary databases (SQLite, EDB), virtual disk containers (VMDK, VHDX), and encrypted or compiled binaries (ELF, PE) where structural format validators are not yet implemented.

---

## 3. CONSTRUCT VALIDITY

Construct validity evaluates whether the mathematical metrics accurately reflect genuine forensic utility and evidential integrity.

### 3.1 Bit-Perfect Recovery ($R_{\text{exact}}$) vs Partial Recovery ($R_{\text{partial}}$)
- **Forensic Distinction:** In digital forensics, a recovered file whose hash does not match the original cannot be cryptographically verified against known hash databases (NSRL/RDS). However, partial recovery ($\ge 50\%$ Longest Common Subsequence) frequently provides crucial investigative intelligence (e.g., readable text inside a PDF or partial thumbnail viewing).
- **Metric Integrity:** ForenSight strictly distinguishes $R_{\text{exact}}$ from $R_{\text{partial}}$, reporting both separately in all publication tables.

### 3.2 False File Merges ($M_{\text{false}}$) as the Critical Forensic Risk
- **Evidential Severity:** In legal proceedings, merging fragments from a suspect's file with an innocent user's document represents a severe false-positive hazard (evidence cross-contamination). ForenSight treats $M_{\text{false}}$ as the primary optimization constraint, penalizing candidate edges that span conflicting format boundaries.

---

## 4. STATISTICAL CONCLUSION VALIDITY

Statistical conclusion validity assesses the mathematical soundness of significance testing and error rate bounds.

### 4.1 Rejection of Fragment-Pair Statistical Units
- **Methodological Flaw in Prior Work:** Many forensic ML papers report $p$-values computed over $10^6$ fragment pairs, claiming astronomical significance ($p < 10^{-50}$). Because pairs sharing fragments are highly correlated, standard errors are artificially compressed by orders of magnitude.
- **Rigorous Standard:** ForenSight computes all bootstrap confidence intervals, Wilcoxon signed-rank tests, and Cliff's delta effect sizes at the **independent file level** ($N = 13$ held-out files, $N = 72$ total corpus).

### 4.2 Non-Parametric Hypothesis Testing & FDR Control
- **Assumption Checking:** File-level recovery metrics violate Gaussianity (bimodal distributions with modes at 0.0 and 1.0).
- **Correction:** We apply two-tailed Wilcoxon signed-rank tests and report non-parametric Cliff's $\delta$. Multiple comparisons across 9 baselines are strictly adjusted using the Benjamini-Hochberg False Discovery Rate procedure ($q = 0.05$).

---

## 5. PRACTICAL OPERATIONAL BOUNDARIES & FAILURE MODES

| Operational Condition | ForenSight Behavior | Recommended Forensic Protocol |
|:---|:---|:---|
| **Contiguous Clean Files** | Bit-perfect recovery ($R_{\text{exact}} = 1.00$) in $<0.1$s | Automated fast-path carving |
| **Interleaved 2--5 Files** | Solves global path-cover ($R_{\text{exact}} = 0.80 - 1.00$) | Full MILP solver with MTZ bounds |
| **Severe Payload Corruption ($>20\%$)** | Accuracy drops to $0.40$; abstention triggers | Manual examiner review of partial streams |
| **Missing Interior Fragments ($>10\%$)** | Exact hash impossible; emits partial chains with gap markers | Mark as non-cryptographic partial evidence |
| **Adversarial Noise / Decoys** | Candidate generator filters $>99\%$ of decoys; abstains on ambiguous noise | Cross-reference with file system timeline |
