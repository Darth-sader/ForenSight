# FORENSIGHT RESEARCH PROJECT
## SCIENTIFIC EVALUATION PROTOCOL (v2.0.0-phase7)

**Target Publication:** *Forensic Science International: Digital Investigation* (FSI:DI)  
**Date:** November 23, 2026  
**Document Status:** Final Approved Protocol  

---

## 1. RESEARCH OBJECTIVE & FORMAL HYPOTHESES

ForenSight investigates whether combining dual-task relationship learning, global constrained discrete optimization, and independent format-aware forensic validation can overcome the failure modes of heuristic file carving under heavy fragmentation, interleaving, missing evidence, and corruption.

### Formal Scientific Hypotheses:
- **Hypothesis 1 (Dual-Relation Superiority):** Independently modeling symmetric same-file co-membership ($P_{\text{same}}$) and asymmetric directed adjacency ($P_{\text{next}}$) significantly reduces false file merges compared to single-relation or next-only models.
- **Hypothesis 2 (Global Optimization vs Greedy Search):** Formulating multi-file reconstruction as a global network path-cover Mixed-Integer Linear Program (MILP) with Miller-Tucker-Zemlin (MTZ) subtour elimination eliminates false file splits and recovers interleaved files that cause greedy beam search to fracture.
- **Hypothesis 3 (Independent Format Constraint Validation):** Applying independent multi-tier format parsers ($L_0$ to $L_4$) prevents machine-learning self-validation and filters out malformed or syntactically invalid paths without biasing ML predictions.
- **Hypothesis 4 (Uncertainty-Aware Selective Abstention):** Calibrated nonconformity scoring and cost-sensitive abstention allow the system to maintain $>95\%$ reconstruction accuracy on accepted files by selectively rejecting ambiguous or corrupt candidates.

---

## 2. DATASET PROVENANCE & LEAKAGE-FREE SPLIT ARCHITECTURE

All experiments strictly adhere to the leakage-free schema defined in Phase 1:
- **Observable Evidence Only:** Inference pipelines receive only `ObservableFragment` attributes (`fragment_id`, `observed_offset`, `fragment_bytes`, `fragment_length`, `allocation_status`, `filesystem_metadata`, `filesystem_region`).
- **Ground-Truth Isolation:** Ground-truth attributes (`source_file_id`, `logical_offset`, `logical_sequence_index`, `original_bytes`, `original_file_hash`) are strictly isolated and accessed exclusively by `EvaluationEngine`.

### Fixed Four-Way Partition Architecture (`split_manifest.json`):
```
All Source Files (N = 50)
├── Training Partition (N = 28 files, 56.0%, 2,058 fragments)
│   └── Used strictly for model parameter fitting and feature scaling.
├── Validation Partition (N = 5 files, 10.0%, 367 fragments)
│   └── Used for hyperparameter tuning, beam search pruning, and cost model thresholding.
├── Calibration Partition (N = 4 files, 8.0%, 290 fragments)
│   └── Used exclusively for probability calibration (Platt/Isotonic) and conformal nonconformity quantiles.
└── Held-Out Test Partition (N = 13 files, 26.0%, 955 fragments)
    └── STRICTLY UNTOUCHED during training, calibration, and hyperparameter selection.
        Evaluated once for final scientific verification.
```

### Source Group Disjointness:
Source files are grouped by structural similarity classes (`doc_group`, `img_group`, `archive_group`, `media_group`). The 4-way split guarantees that related files from the same generator configuration are assigned to the same partition, preventing cross-file representation leakage.

---

## 3. FORMAL EVALUATION METRIC DEFINITIONS

All metrics define explicit mathematical numerators and denominators. Partial recovery is strictly distinguished from exact byte-perfect recovery.

### 3.1 File-Level Metrics
Let $\mathcal{F}_{\text{gt}} = \{F_1, \dots, F_M\}$ be the set of ground-truth files, and $\mathcal{R} = \{R_1, \dots, R_K\}$ be the set of reconstructed files emitted by an algorithm.

1. **Exact Reconstruction Rate ($R_{\text{exact}}$):**
   $$R_{\text{exact}} = \frac{1}{|\mathcal{F}_{\text{gt}}|} \sum_{F \in \mathcal{F}_{\text{gt}}} \mathbf{1}\left\{\exists R \in \mathcal{R} : \text{SHA256}(R) == \text{SHA256}(F)\right\}$$
   *Requires exact bit-for-bit SHA-256 cryptographic hash equality. Partial or padded matches evaluate to 0.*

2. **Partial Reconstruction Rate ($R_{\text{partial}}$):**
   $$R_{\text{partial}} = \frac{1}{|\mathcal{F}_{\text{gt}}|} \sum_{F \in \mathcal{F}_{\text{gt}}} \mathbf{1}\left\{\frac{\text{LCS}(\text{seq}(R^*), \text{seq}(F))}{|\text{seq}(F)|} \ge 0.50 \text{ and } \text{SHA256}(R^*) \neq \text{SHA256}(F)\right\}$$
   Where $R^*$ is the best-matching reconstructed file by Longest Common Subsequence (LCS).

3. **Byte Recovery Ratio ($B_{\text{ratio}}$):**
   $$B_{\text{ratio}} = \frac{\sum_{F \in \mathcal{F}_{\text{gt}}} \max_{R \in \mathcal{R}} \text{LCS\_Bytes}(R, F)}{\sum_{F \in \mathcal{F}_{\text{gt}}} \text{Length}(F)}$$

4. **False File Merge Rate ($M_{\text{false}}$):**
   $$M_{\text{false}} = \frac{|\{R \in \mathcal{R} : |\{F \in \mathcal{F}_{\text{gt}} : \text{Fragments}(F) \cap \text{Fragments}(R) \neq \emptyset\}| \ge 2\}|}{|\mathcal{R}|}$$
   *Measures the proportion of reconstructed files that erroneously concatenate fragments from two or more distinct original files.*

5. **False File Split Rate ($S_{\text{false}}$):**
   $$S_{\text{false}} = \frac{|\{F \in \mathcal{F}_{\text{gt}} : |\{R \in \mathcal{R} : \text{Fragments}(F) \cap \text{Fragments}(R) \neq \emptyset\}| \ge 2\}|}{|\mathcal{F}_{\text{gt}}|}$$
   *Measures the proportion of original files that are fractured across two or more distinct reconstructed files.*

### 3.2 Fragment-Level Metrics
1. **Fragment Placement Accuracy ($P_{\text{frag}}$):**
   $$P_{\text{frag}} = \frac{1}{N_{\text{gt}}} \sum_{F \in \mathcal{F}_{\text{gt}}} \sum_{i=0}^{|F|-1} \mathbf{1}\left\{\text{Fragment at position } i \text{ in best matching } R \text{ matches position } i \text{ in } F\right\}$$

2. **Successor Precision ($P_{\text{succ}}$) & Recall ($R_{\text{succ}}$):**
   $$P_{\text{succ}} = \frac{\text{Number of correct directed transitions } (u \to v) \text{ in } \mathcal{R}}{\text{Total directed transitions in } \mathcal{R}}$$
   $$R_{\text{succ}} = \frac{\text{Number of correct directed transitions } (u \to v) \text{ in } \mathcal{R}}{\text{Total ground-truth successor transitions in } \mathcal{F}_{\text{gt}}}$$

### 3.3 Calibration & Uncertainty Metrics
1. **Expected Calibration Error (ECE):**
   $$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$
   Computed across $M=10$ equal-width probability bins on the test partition.
2. **Brier Score:**
   $$\text{Brier} = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2$$

---

## 4. BASELINE SPECIFICATIONS & EXECUTION PROTOCOLS

To ensure fair, scientifically rigorous comparisons, ForenSight is benchmarked against 9 distinct baseline configurations representing the state of the art in file carving and graph reconstruction:

1. **PhotoRec v7.2 Baseline (Header/Footer Heuristic Carver):**
   - *Protocol:* Standard carving mode using header magic signatures and footer terminating patterns.
   - *Underlying Assumption:* Assumes contiguous allocation. Incapable of de-interleaving fragmented clusters.
   - *Input:* Contiguous raw disk byte stream.
2. **Scalpel v2.0 Baseline (Rule-Based Carver):**
   - *Protocol:* Standard configuration with defined header/footer rules and maximum extract size limits.
   - *Underlying Assumption:* Fixed-window contiguous extraction without relational inference.
3. **Original ForenSight Baseline (v1 Inception):**
   - *Protocol:* Next-only probability reuse for both same-file and next-fragment tasks, unconstrained heuristic beam search ($W=5$), no MTZ cycle constraints, no mutual exclusion.
4. **Leakage-Free Pairwise Baseline:**
   - *Protocol:* Random Forest classifier operating on leakage-free features, greedy succession.
5. **Same-File-Only Reconstruction (Variant B):**
   - *Protocol:* Connected components clustering using $P_{\text{same}}$ without directed adjacency ordering.
6. **Next-Fragment-Only Reconstruction (Variant A):**
   - *Protocol:* Pure directed path-following using $P_{\text{next}}$ without co-membership verification.
7. **Dual-Relation Independent Reconstruction (Variant C):**
   - *Protocol:* Symmetric $P_{\text{same}}$ filtering combined with directed $P_{\text{next}}$ beam search.
8. **Multi-Task Dual Model (Variant D):**
   - *Protocol:* Shared-backbone neural network predicting both relationships simultaneously.
9. **GNN-Refined Reconstruction (GraphSAGE / GAT):**
   - *Protocol:* 2-layer Graph Neural Network message passing over candidate graph edges.
10. **Full ForenSight Framework (Method D):**
    - *Protocol:* Global constrained MILP optimization with MTZ subtour elimination, format boundary guards, missing gap variables, isotonic probability calibration, and cost-sensitive abstention.

---

## 5. STATISTICAL METHODOLOGY

> [!IMPORTANT]
> **Statistical Unit of Analysis: Source Files ($N=13$ Test Files, $N=50$ Corpus):**  
> We explicitly reject the invalid practice of computing statistical significance over millions of fragment pairs. Pairs sharing fragments violate independence and artificially deflate standard errors.

1. **Bootstrap Confidence Intervals:**
   - 1,000 bootstrap resamples computed at the **file level**.
   - 95% Confidence Intervals reported as $[Q_{0.025}, Q_{0.975}]$.
2. **Paired Statistical Hypothesis Testing:**
   - **Wilcoxon Signed-Rank Test:** Two-tailed non-parametric paired test comparing file-level byte recovery and placement accuracy between ForenSight Full and each baseline.
   - **McNemar's Test:** Exact paired test for binary exact reconstruction outcomes ($R_{\text{exact}} \in \{0, 1\}$).
3. **Effect Size Measures:**
   - **Cliff's Delta ($\delta$):** Non-parametric effect size for ordinal/continuous metrics:
     $$\delta = \frac{\#(\text{ForenSight} > \text{Baseline}) - \#(\text{ForenSight} < \text{Baseline})}{N_1 \cdot N_2}$$
     Interpreted as: Negligible ($|\delta| < 0.147$), Small ($0.147 \le |\delta| < 0.33$), Medium ($0.33 \le |\delta| < 0.474$), Large ($|\delta| \ge 0.474$).
4. **Multiple Comparison Handling:**
   - Family-wise error rate controlled via the **Benjamini-Hochberg False Discovery Rate (FDR)** procedure at $q = 0.05$.
