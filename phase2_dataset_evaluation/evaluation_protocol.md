# Digital Forensic File Reconstruction Evaluation Protocol

**ForenSight Research Framework — Phase 2**  
**Document Revision:** `v2.0.0-phase2`  
**Standard:** Double-Blind Ground-Truth Forensic Evaluation Protocol  

---

## 1. Scope & Objective

This protocol governs the experimental evaluation of machine learning models, graph inference algorithms, and forensic carving methods within the ForenSight framework.

The primary objective is to evaluate whether learned representations, graph neural networks, and forensic constraints can reliably reconstruct fragmented digital files under real-world degradation without ground-truth leakage or statistical contamination.

---

## 2. Experimental Conditions

All reconstruction algorithms must be evaluated under two standardized forensic conditions:

### Condition A: Metadata-Independent Mode (Carved Unallocated Evidence)
- **Forensic Context**: Raw unallocated drive slack or carved block stream with corrupted or missing filesystem allocation tables.
- **Observable Input**: Raw fragment byte payloads ($b_i$) and generic block IDs.
- **Physical Offsets**: Uniformly set to 0. No physical proximity hints or cluster offsets are available.
- **Purpose**: Establishes the true capability of content-based, structural, and byte-level modeling.

### Condition B: Observed-Layout Mode (Forensic Disk Image Carving)
- **Forensic Context**: Acquisition of a complete raw storage volume (`.raw`, `.dd`).
- **Observable Input**: Physical sector LBA, allocation status (allocated vs. unallocated bitmap), filesystem region (system vs. data).
- **Physical Offsets**: Derived strictly from observed physical disk LBAs. Logical source file offsets remain strictly forbidden.
- **Purpose**: Evaluates candidate generation efficiency when physical cluster locality is available alongside payload features.

---

## 3. Strict Isolation Rules

1. **Schema Validation Gate**: Prior to inference, every evidence record must pass through `validate_observable_record()`. The presence of any ground-truth attribute (`source_file`, `logical_offset`, `original_position`, `is_corrupted`, `is_missing`) triggers an immediate `SchemaLeakageError` and voids the benchmark.
2. **Preprocessing Isolation**: Any scaler, normalizer, imputer, or dimensionality reduction transformation must be fitted strictly on the training partition (`X_train`). Test and holdout partitions must be evaluated strictly using frozen transformation parameters.
3. **Partition Boundary Integrity**: Test and holdout files must never contribute to hyperparameter optimization, threshold calibration, or feature selection.
4. **Separation of External and Synthetic Benchmarks**: Benchmarks conducted on external corpora (e.g. Govdocs1, DFRWS 2006) must be stored and reported in separate artifact files (`results/external_eval_*.json`) and never mixed with synthetic corpus results.

---

## 4. Evaluated Metrics Hierarchy

Reconstruction performance is evaluated across four tiers:

```
+-----------------------------------------------------------------------------------+
| 1. Exact File Recovery (Gold Standard)                                            |
|    - Exact Byte-Perfect Reconstruction Rate (SHA-256 Match)                       |
|    - Partial Sequence Recovery Rate (LCS >= 0.50)                                 |
|    - Byte Recovery Ratio (Correct contiguous recovered bytes / Total GT bytes)   |
+-----------------------------------------------------------------------------------+
| 2. Forensic Integrity & Error Rates                                               |
|    - False File Merge Rate (Assembling fragments from multiple files)             |
|    - False File Split Rate (Fragmenting a single file across multiple outputs)    |
|    - False Reconstruction Rate (Emitting invalid / non-ground-truth files)         |
+-----------------------------------------------------------------------------------+
| 3. Fragment Adjacency & Candidate Retrieval                                       |
|    - Successor Precision, Recall, and F1 Score                                    |
|    - Successor Recall@K (K = 1, 3, 5, 10)                                         |
|    - Fragment Placement Accuracy                                                  |
|    - Duplicate Fragment Assignment Rate & Unused Fragment Rate                    |
+-----------------------------------------------------------------------------------+
| 4. Computational Efficiency                                                       |
|    - Total Execution Runtime (seconds) & Search Time per File (ms)                |
|    - Peak Memory Usage (MB RSS)                                                   |
|    - Candidate Graph Node & Edge Cardinality                                      |
|    - Total Pairwise Inference Evaluations                                         |
+-----------------------------------------------------------------------------------+
```

---

## 5. Statistical Significance Testing Protocol

To prove that a proposed reconstruction method (e.g. GNN + CP-SAT) genuinely outperforms baselines (e.g. pairwise greedy or beam search), evaluation must include:

### 5.1. Binary Classification & Placement Significance (McNemar's Test)
For pairwise transition accuracy and fragment placement:
$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$$
Where:
- $b$: Transitions correctly predicted by Method B but missed by Method A.
- $c$: Transitions correctly predicted by Method A but missed by Method B.
- Evaluated at significance threshold $\alpha = 0.01$.

### 5.2. File-Level Recovery Significance (Wilcoxon Signed-Rank Test)
For exact recovery rate and byte recovery ratio across the 11 scenario instances:
- Paired differences $d_i = \text{Score}_B(i) - \text{Score}_A(i)$ across all 11 standardized scenarios.
- Wilcoxon signed-rank test on ranked non-zero differences to reject the null hypothesis of equal median recovery.

---

## 6. Known-Answer Validation Protocol

Before any experimental run is certified:
1. The test suite `phase2_dataset_evaluation/evaluator_tests.py` must be executed.
2. The suite must verify:
   - 100% exact recovery on perfect inputs.
   - Exact recovery remains 0% on partial or missing fragment inputs.
   - False merge rate evaluates to 1.0 on chimeric files.
   - Duplicate fragment assignment is accurately counted.
   - Non-matching ordering evaluates to 0% placement accuracy.
3. Test runs must achieve 100% pass rate before logging benchmark results to the database.
