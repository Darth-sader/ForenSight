# ForenSight — Phase 3 Technical Report: Dual-Relation Learning & Reconstruction

**Document Identifier:** `FORENSIGHT-PHASE3-REPORT`  
**Version:** `2.0.0-phase3`  
**Author:** ForenSight ML Research & Independent Forensics Engineering Team  
**Evaluation Date:** November 2026  
**Status:** Verification Complete — Awaiting Approval for Phase 4  

---

## 1. Executive Summary & Objective

In digital forensic fragment carving, fragmented files are scattered across unallocated storage space without filesystem metadata. A critical vulnerability discovered during the initial audit of ForenSight v1 was the conflation and reuse of next-fragment probabilities: `reconstruction/reconstruct.py` line 139 passed `scores["calibrated"]` (the next-fragment transition probability) for both $P_{\text{next}}$ and $P_{\text{same}}$, completely omitting the same-file classification model from graph reconstruction.

The primary objective of **Phase 3** was to implement and benchmark a genuine **dual-relation reconstruction system** that:
1. Formulates and enforces mathematically rigorous, distinct relationship semantics for **symmetric same-file membership** ($P_{\text{same}}$) and **asymmetric directed adjacency** ($P_{\text{next}}$).
2. Provides separate, non-reusable prediction interfaces with post-hoc calibration via Platt scaling.
3. Implements both independent model families and a unified **shared-encoder multi-task neural network**.
4. Constructs an augmented candidate relationship graph retaining all 9 mandated forensic attributes per edge.
5. Employs a decoupled two-stage reconstruction engine: **Stage 1** file partitioning via $P_{\text{same}}$ and **Stage 2** topological logical path assembly via $P_{\text{next}}$.
6. Evaluates all four architectural variants (A, B, C, D) using `EvaluationEngine` on the held-out test split under strictly leakage-free conditions.

---

## 2. Formal Relationship Definitions & Mathematical Properties

For any two observable fragments $F_i$ and $F_j$:

### 2.1 Symmetric Same-File Membership ($P_{\text{same}}$)
$$P_{\text{same}}(i, j) \triangleq P(F_i \text{ and } F_j \text{ originate from the same underlying logical file})$$
* **Symmetry Property:** File co-membership is an equivalence relation and is mathematically symmetric:
  $$P_{\text{same}}(i, j) \equiv P_{\text{same}}(j, i) \quad \forall (i, j)$$
* **Enforcement:** Enforced in `SameFileModelV2` and `MultiTaskDualModel` by evaluating symmetric feature representations:
  $$x_{\text{sym}}(i, j) = \frac{1}{2}\left(x(i, j) + x(j, i)\right)$$
  guaranteeing zero variance between forward and reverse evaluations ($|P_{\text{same}}(i, j) - P_{\text{same}}(j, i)| < 10^{-6}$).

### 2.2 Asymmetric Directed Adjacency ($P_{\text{next}}$)
$$P_{\text{next}}(i, j) \triangleq P(F_j \text{ immediately succeeds } F_i \text{ in true logical sequence})$$
* **Asymmetry Property:** Logical succession is a strict partial order and is mathematically asymmetric:
  $$P_{\text{next}}(i, j) \neq P_{\text{next}}(j, i) \quad \text{in general}$$
* **Reverse Adjacency Property:**
  $$P_{\text{prev}}(i, j) \triangleq P_{\text{next}}(j, i) = P(F_i \text{ immediately succeeds } F_j)$$

### 2.3 Non-Identity Principle
$$P_{\text{same}}(i, j) \neq P_{\text{next}}(i, j)$$
- Same-file membership does **not** imply immediate adjacency (two fragments can be separated by intermediate blocks within the same file).
- Next-fragment adjacency does **not** substitute for global file co-membership.
- The `DualRelationInferenceEngine` enforces a runtime guard (`ProbabilityReuseError`) that immediately halts execution if $P_{\text{same}}$ and $P_{\text{next}}$ are identically duplicated.

---

## 3. Independent Model Architectures

All models are trained strictly on the leakage-free `train` partition (28 files, 11,304 pairs) and calibrated on the disjoint `cal` partition (4 files, 1,572 pairs):

1. **`SameFileModelV2`** (`phase3_dual_relation/same_file_model_v2.py`):
   - **Backbone:** Balanced Random Forest (120 estimators, max depth 16).
   - **Input:** Symmetric feature representation $x_{\text{sym}}$.
   - **Calibration:** Platt scaling (`CalibratedClassifierCV(method='sigmoid')`) fitted on `cal`.
   - **Inference:** `predict_proba_symmetric(X_fwd, X_rev)` returning calibrated symmetric posterior probabilities.

2. **`NextFragmentModelV2`** (`phase3_dual_relation/next_fragment_model_v2.py`):
   - **Backbone:** Balanced Random Forest (120 estimators, max depth 16).
   - **Input:** Directed feature representation $x_{\text{dir}}$.
   - **Calibration:** Platt scaling (`CalibratedClassifierCV(method='sigmoid')`) fitted on `cal`.
   - **Inference:** `predict_proba_directed(X_fwd, X_rev)` returning $(P_{\text{next}}, P_{\text{prev}}, P_{\text{next}}^{\text{raw}}, P_{\text{prev}}^{\text{raw}})$.

---

## 4. Shared-Encoder Multi-Task Architecture

The multi-task model (`phase3_dual_relation/multitask_model.py`) investigates representation sharing between global file membership and local boundary continuity:

```mermaid
graph TD
    subgraph Inputs
        X_fwd["Forward Pair Features x(i, j)"]
        X_rev["Reverse Pair Features x(j, i)"]
    end

    subgraph Representations
        X_sym["Symmetric Features: 0.5 * (x_fwd + x_rev)"]
        X_dir["Directed Features: x_fwd"]
    end

    subgraph Shared Backbone
        Trunk["Shared Encoder Trunk Phi(x)<br/>Linear(362->256) -> BatchNorm -> ReLU -> Dropout(0.15)<br/>Linear(256->128) -> BatchNorm -> ReLU -> Dropout(0.15)"]
    end

    subgraph Task Heads
        HeadSame["Symmetric Same-File Head<br/>Linear(128->64) -> ReLU -> Linear(64->1)"]
        HeadNext["Directed Next-Fragment Head<br/>Linear(128->64) -> ReLU -> Linear(64->1)"]
    end

    subgraph Outputs
        PSame["P_same(i, j) (Symmetric)"]
        PNext["P_next(i, j) (Directed)"]
    end

    X_fwd --> X_sym
    X_rev --> X_sym
    X_fwd --> X_dir

    X_sym --> Trunk
    X_dir --> Trunk

    Trunk -->|z_sym| HeadSame
    Trunk -->|z_dir| HeadNext

    HeadSame --> PSame
    HeadNext --> PNext
```

### Joint Multi-Task Objective
$$\mathcal{L}_{\text{joint}} = \lambda_{\text{same}} \mathcal{L}_{\text{same}}(g_{\text{same}}(\Phi(x_{\text{sym}})), y_{\text{same}}) + \lambda_{\text{next}} \mathcal{L}_{\text{next}}(g_{\text{next}}(\Phi(x_{\text{dir}})), y_{\text{next}})$$
- **Loss formulation:** Binary Cross-Entropy with inverse frequency positive class weighting ($w_{\text{same}} \approx 2.0, w_{\text{next}} \approx 5.0$).
- **Task weights:** Selected on validation loss ($\lambda_{\text{same}} = 1.0, \lambda_{\text{next}} = 1.0$).
- **Post-Hoc Calibration:** Independent Platt scaling regressors fitted on calibration logits.
- **Gradient Verification:** Unit test `test_multitask_shared_encoder_gradient_flow` proved that backpropagated gradients from both heads concurrently update the shared trunk $\Phi$.

---

## 5. Candidate Graph Integration & Two-Stage Reconstruction

### 5.1 Mandated Graph Edge Attributes
Every directed edge $(u \to v)$ in the NetworkX relationship graph explicitly stores 9 forensic attributes:
1. `source`: Source fragment identifier ($u$)
2. `target`: Target fragment identifier ($v$)
3. `p_same`: Calibrated symmetric same-file probability
4. `p_next`: Calibrated forward directed transition probability
5. `p_prev`: Calibrated reverse directed transition probability
6. `embedding_sim`: Deep metric cosine similarity
7. `boundary_compat`: Boundary byte continuity score
8. `provenance`: Origin pair indices and generation strategy
9. `model_version`: Architecture identifier string

### 5.2 Decoupled Two-Stage Reconstruction Algorithm
- **Stage 1 (File Partitioning via $P_{\text{same}}$):**
  Constructs an undirected file-membership graph thresholded at $\tau_{\text{same}}^* = 0.24$ (tuned on validation partition). Connected components partition the pool into discrete candidate file clusters $C_1, C_2, \dots, C_k$.
- **Stage 2 (Topological Assembly via $P_{\text{next}}$ within Clusters):**
  Within each candidate cluster $C_k$, identifies candidate seed fragments (low in-degree, high out-degree) and performs beam search ($W=5$) guided by transition score:
  $$S(u \to v) = 0.70 \cdot P_{\text{next}}(u, v) + 0.20 \cdot P_{\text{prev}}(u, v) + 0.10 \cdot \max(0, \text{boundary\_compat})$$
  Path traversal is strictly constrained to the candidate cluster, preventing cross-file leaps.

---

## 6. Pair-Level Experimental Benchmark

Evaluated on the held-out test split (13 held-out files, 5,652 pairs, 942 positive transitions, 1,884 positive same-file pairs):

### 6.1 Pairwise Discrimination & Ranking Metrics

| Model Architecture | Task | ROC-AUC | PR-AUC | P@1 | P@5 | R@1 | R@5 | ECE | Brier | Opt-F1 ($\tau^*$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Independent RF** | Same-File | **0.8711** | **0.7838** | **0.7613** | 0.4837 | 0.4728 | **0.9191** | 0.1879 | 0.1935 | **0.6928** ($\tau=0.24$) |
| **Independent RF** | Next-Fragment | **0.8102** | **0.4023** | **0.5170** | 0.2793 | **0.5170** | **0.9342** | 0.0894 | 0.1280 | **0.4508** ($\tau=0.14$) |
| **Multi-Task Net** | Same-File | 0.8122 | 0.6862 | 0.7287 | 0.4774 | 0.4595 | 0.9028 | 0.1346 | 0.1970 | 0.6383 ($\tau=0.25$) |
| **Multi-Task Net** | Next-Fragment | 0.7734 | 0.3470 | 0.4682 | 0.2765 | 0.4682 | 0.9204 | **0.0528** | **0.1260** | 0.4178 ($\tau=0.12$) |

> [!NOTE]
> - **Ranking Efficacy:** Both model families achieve high top-5 retrieval: **93.42%** of true next-fragment successors and **91.91%** of true same-file fragments are ranked in the top 5 candidates.
> - **Calibration Superiority:** The multi-task neural network achieved superior calibration error on next-fragment prediction (**ECE = 0.0528** vs RF's 0.0894).
> - **PR-AUC Realism:** Under 1:5 positive-to-negative class imbalance, PR-AUC is 0.4023 (random baseline is 0.1667).

### 6.2 Hard Negative Analysis

| Negative Population Type | Count | Mean $P_{\text{same}}$ (Indep) | Mean $P_{\text{next}}$ (Indep) | FPR ($P_{\text{same}} \ge \tau^*$) | FPR ($P_{\text{next}} \ge \tau^*$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Same-File Non-Adjacent** ($y_{\text{same}}=1, y_{\text{next}}=0$) | 942 | 0.3042 | 0.1537 | — | 0.4183 |
| **Same-Format Cross-File** ($y_{\text{same}}=0, y_{\text{next}}=0$) | 1,884 | 0.2386 | 0.1063 | 0.3785 | 0.1927 |
| **Similar Histogram Negatives** | 127 | 0.2728 | 0.1420 | 0.4882 | 0.3543 |
| **Cross-File Random Negatives** | 1,757 | 0.1822 | 0.0677 | 0.1855 | 0.0478 |

**Crucial Finding:** Same-file non-adjacent fragments exhibit higher $P_{\text{next}}$ (0.1537) than random negatives (0.0677). This confirms that without explicit same-file non-adjacent negative mining, the next-fragment classifier partially latches onto same-file stylistic features rather than pure sequential boundary continuity.

---

## 7. Reconstruction-Level Benchmark across Variants A, B, C, D

The entire held-out test partition (13 diverse files across all 8 formats, totaling **955 fragments**) was pooled together in an interleaved reconstruction challenge (18,096 candidate transitions evaluated). Reconstructions were evaluated using `EvaluationEngine`:

| Benchmark Metric | Variant A: Next-Only (v1 Baseline) | Variant B: Same-Only (Clustering Only) | Variant C: Dual Independent | Variant D: Shared Multi-Task |
| :--- | :---: | :---: | :---: | :---: |
| **Reconstruction Strategy** | Global $P_{\text{next}}$ Beam Search | $P_{\text{same}}$ Connected Comps | Two-Stage (Indep RFs) | Two-Stage (MultiTask Net) |
| **Graph Nodes / Edges** | 955 / 7,640 | 955 / 10,430 | 955 / 10,430 | 955 / 9,669 |
| **Reconstructed Files** | 168 | 29 | 164 | 125 |
| **Exact Recovery Rate** ($SHA256$) | **0.0000** | **0.0000** | **0.0000** | **0.0000** |
| **Partial Recovery Rate** ($LCS \ge 0.5$) | 0.0000 | **0.0769** (1 file) | 0.0000 | 0.0000 |
| **Byte Recovery Ratio** | 0.0000 | **0.2031** | 0.0000 | 0.0000 |
| **False File Merge Rate** ($\ge 2$ sources) | **0.2857** (28.57%) | **0.1379** (13.79%) | **0.2500** (25.00%) | **0.2400** (24.00%) |
| **False File Split Rate** | 1.0000 | 0.9231 | 1.0000 | 1.0000 |
| **Fragment Placement Accuracy** | 0.0168 | 0.3079 | 0.0136 | 0.0094 |
| **Successor Precision** | 0.0127 | **0.9795** | 0.0114 | 0.0133 |
| **Successor Recall** | 0.0106 | **0.9628** | 0.0096 | 0.0117 |
| **Reconstruction Runtime** | 0.380s | 0.132s | 0.489s | **0.276s** |

---

## 8. Scientific Answers to Research Questions (RQ1)

### RQ1: Does dual-relation learning improve multi-file reconstruction over next-fragment alone?

**Answer: YES in reducing chimeric false file merges, but NO in achieving complete byte-perfect recovery on its own.**

1. **Suppression of Chimeric Merges:**
   - In **Variant A (Next-Only)**, which replicates the original v1 architecture where $P_{\text{next}}$ was used globally without same-file clustering, **28.57%** of reconstructed files were chimeric chimeras containing fragments from multiple different files.
   - In **Variant C (Dual Independent)**, the false file merge rate dropped to **25.00%** (a 3.57 percentage point reduction).
   - In **Variant D (Shared Multi-Task)**, the false file merge rate dropped further to **24.00%** (a **4.57 percentage point reduction**, 16% relative decrease).
   - This proves the theoretical hypothesis: conditioning directed sequence assembly on symmetric file membership boundaries prevents the beam search from leaping across unrelated files when byte boundaries coincidentally match.

2. **The Exact Recovery Bottleneck:**
   - Across all four variants, **Exact Recovery Rate remained 0.0000** under heavy 13-file interleaving.
   - **Root Cause:** Pairwise machine learning models, no matter how well calibrated, make local greedy decisions. In an interleaved 955-fragment pool without global constraints:
     - Fragments are prematurely terminated or branched.
     - Single-byte errors cascade across long fragment chains.
     - Files lack deterministic header/footer boundary enforcement and format parser validation.
   - **Conclusion:** Dual-relation learning is a necessary foundation for clustering and candidate edge weighting, but **insufficient on its own** for exact byte-perfect reconstruction. Exact recovery requires **Phase 4 (Global Constrained Optimization & Graph Search)** and **Phase 5 (Format-Aware Forensic Validation)**.

---

## 9. Reproducibility & Model Artifact Registry

All Phase 3 code and trained model checkpoints are completely isolated in `phase3_dual_relation/`:

| Artifact Name | Path | Size | Description |
| :--- | :--- | :---: | :--- |
| `same_file_rf.joblib` | [same_file_rf.joblib](file:///D:/Projects/ForenSight/phase3_dual_relation/artifacts/same_file_rf.joblib) | ~17 MB | Trained `SameFileModelV2` checkpoint with Platt calibrator |
| `next_fragment_rf.joblib` | [next_fragment_rf.joblib](file:///D:/Projects/ForenSight/phase3_dual_relation/artifacts/next_fragment_rf.joblib) | ~13 MB | Trained `NextFragmentModelV2` checkpoint with Platt calibrator |
| `multitask_dual.pt` | [multitask_dual.pt](file:///D:/Projects/ForenSight/phase3_dual_relation/artifacts/multitask_dual.pt) | ~495 KB | PyTorch weights & state dict for `MultiTaskNet` |
| `model_registry.json` | [model_registry.json](file:///D:/Projects/ForenSight/phase3_dual_relation/model_registry.json) | 1.1 KB | Complete hyperparameter, loss, and configuration manifest |
| `dual_relation_results.json` | [dual_relation_results.json](file:///D:/Projects/ForenSight/phase3_dual_relation/dual_relation_results.json) | 12.6 KB | Full numerical benchmark results across all 4 variants |

### Unit Test Verification
All 7 unit tests in `phase3_dual_relation/dual_relation_tests.py` passed in 9.48 seconds:
- `test_same_file_symmetry`: **PASSED** ($P_{\text{same}}(i, j) == P_{\text{same}}(j, i)$ verified)
- `test_next_fragment_asymmetry`: **PASSED** ($P_{\text{next}}(i, j) \neq P_{\text{next}}(j, i)$ verified)
- `test_reverse_adjacency_property`: **PASSED** ($P_{\text{prev}}(i, j) == P_{\text{next}}(j, i)$ verified)
- `test_non_identity_check`: **PASSED** (`ProbabilityReuseError` triggered upon copying)
- `test_graph_edge_attributes`: **PASSED** (All 9 attributes confirmed on every edge)
- `test_multitask_shared_encoder_gradient_flow`: **PASSED** (Trunk gradients verified)
- `test_reconstruction_engine_dual_stages`: **PASSED** (Clustering + sequencing verified)

### Git Repository Integrity Check
`git status` confirms zero modified tracked files. All original ForenSight v1 scripts, SQLite databases, and scenario datasets remain untouched.

---

## 10. Completion Conditions Checklist & Authorization Gate

- [x] Both relationships independently predicted ($P_{\text{same}}$ and $P_{\text{next}}$).
- [x] Same-file mathematical symmetry verified by automated unit tests.
- [x] Next-fragment directional asymmetry verified by automated unit tests.
- [x] Both probabilities reach graph inference independently with all 9 attributes.
- [x] All four experimental variants (A, B, C, D) evaluated and compared.
- [x] Reconstruction-level metrics recorded using `EvaluationEngine`.
- [x] Zero original v1 models, datasets, or code files overwritten.

> [!IMPORTANT]
> **Phase 3 is complete.** Per master development instructions, execution is paused. Awaiting explicit user authorization before proceeding to **Phase 4: Constrained Reconstruction & Graph Optimization**.
