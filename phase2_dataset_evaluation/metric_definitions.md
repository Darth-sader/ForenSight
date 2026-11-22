# Forensic File Reconstruction Metric Definitions & Formal Standards

**ForenSight Research Framework — Phase 2**  
**Document Revision:** `v2.0.0-phase2`  
**Evaluation Standard:** Strict Ground-Truth Reconstruction Verification  

---

## 1. Overview & Evaluation Principles

In digital forensic file reconstruction research, evaluation must distinguish between:
1. **Exact Byte-Perfect Reconstruction**: Cryptographically verified via SHA-256 hash match against the original source file.
2. **Partial Reconstruction**: Useful contiguous content recovered, but hash does not match original (e.g. due to missing blocks or corruption).
3. **False Recovery / Forensically Invalid Claims**: Merging fragments from unrelated files, or emitting non-functional corrupt files.

Every metric in the ForenSight Evaluation Engine is formalized below with its exact mathematical numerator, denominator, and edge-case behavior.

---

## 2. Fragment-Level Metrics

### 2.1. Fragment Placement Accuracy ($\text{Acc}_{\text{place}}$)
Measures the proportion of fragments positioned at their exact ground-truth logical index within their correct source file.
$$\text{Acc}_{\text{place}} = \frac{\sum_{i=1}^{N_{\text{recon}}} \mathbb{I}(\text{Frag}_i \in \text{File}_{\text{true}} \land \text{pos}_i == \text{pos}_{\text{true}})}{N_{\text{GT\_frags}}}$$
- **Numerator**: Count of fragments placed at the exact ground-truth sequence position.
- **Denominator**: Total number of fragments in ground-truth corpus ($N_{\text{GT\_frags}}$).
- **Edge cases**: If $N_{\text{GT\_frags}} = 0$, evaluates to $0.0$. Fragments from false merges or misplaced indices count as $0$.

### 2.2. Successor Precision ($\text{Prec}_{\text{succ}}$)
Measures the accuracy of reconstructed fragment transitions ($f_a \to f_b$).
$$\text{Prec}_{\text{succ}} = \frac{\sum_{(f_a, f_b) \in E_{\text{recon}}} \mathbb{I}(\text{File}(f_a) == \text{File}(f_b) \land \text{Seq}(f_b) == \text{Seq}(f_a) + 1)}{|E_{\text{recon}}|}$$
- **Numerator**: Count of valid consecutive transitions in reconstructed files.
- **Denominator**: Total number of fragment transitions across all reconstructed files ($|E_{\text{recon}}| = \sum_{F} (\text{len}(F) - 1)$).

### 2.3. Successor Recall ($\text{Rec}_{\text{succ}}$)
Measures the proportion of true consecutive fragment transitions successfully recovered.
$$\text{Rec}_{\text{succ}} = \frac{\sum_{(f_a, f_b) \in E_{\text{recon}}} \mathbb{I}(\text{File}(f_a) == \text{File}(f_b) \land \text{Seq}(f_b) == \text{Seq}(f_a) + 1)}{|E_{\text{GT}}|}$$
- **Numerator**: Count of valid consecutive transitions in reconstructed files.
- **Denominator**: Total ground-truth consecutive transitions across all files ($|E_{\text{GT}}| = \sum_{F \in \text{GT}} (\text{len}(F) - 1)$).

### 2.4. Successor Recall@K ($\text{Rec@K}$)
Measures whether the true immediate successor of fragment $f_a$ appears in the top-$K$ candidate edges ranked by the relationship model.
$$\text{Rec@K} = \frac{1}{|V_{\text{anchors}}|} \sum_{f_a \in V_{\text{anchors}}} \mathbb{I}(f_{\text{true\_next}}(f_a) \in \text{TopK}(f_a))$$
- Evaluated for $K \in \{1, 3, 5, 10\}$. Demonstrates whether search algorithms will encounter the true edge within beam width $K$.

### 2.5. Duplicate Fragment Assignment Rate ($\text{Rate}_{\text{dup}}$)
Measures the fraction of fragments assigned to multiple reconstructed files or multiple positions.
$$\text{Rate}_{\text{dup}} = \frac{N_{\text{total\_assigned}} - |V_{\text{unique\_assigned}}|}{N_{\text{total\_assigned}}}$$
- **Denominator**: Total fragment tokens across all reconstructed files. Evaluates to $0.0$ for physically valid disjoint carvings.

### 2.6. Unused Fragment Rate ($\text{Rate}_{\text{unused}}$)
Measures the proportion of evidence fragments left unassigned in the fragment pool.
$$\text{Rate}_{\text{unused}} = \frac{|V_{\text{pool}} \setminus V_{\text{assigned}}|}{|V_{\text{pool}}|}$$

---

## 3. File-Level Metrics

### 3.1. Exact Byte-Perfect Reconstruction Rate ($\text{Rate}_{\text{exact}}$)
The gold standard for digital forensic recovery: exact cryptographic SHA-256 hash match against the original unfragmented file.
$$\text{Rate}_{\text{exact}} = \frac{\sum_{F_{\text{gt}} \in \text{GT}} \mathbb{I}(\exists F_{\text{recon}} : \text{SHA256}(F_{\text{recon}}) == \text{SHA256}(F_{\text{gt}}))}{|\text{GT}|}$$
- **Rule**: Partial reconstruction, even at 99.9% byte match, is **strictly excluded** from this numerator.

### 3.2. Partial Reconstruction Rate ($\text{Rate}_{\text{partial}}$)
Measures files that failed exact hash match but recovered at least 50% of the original fragment sequence via Longest Common Subsequence (LCS).
$$\text{Rate}_{\text{partial}} = \frac{\sum_{F_{\text{gt}} \in \text{GT}} \mathbb{I}(\text{LCS}(F_{\text{recon}}, F_{\text{gt}}) \ge 0.5 \times |F_{\text{gt}}| \land \neg \text{Exact}(F_{\text{gt}}))}{|\text{GT}|}$$

### 3.3. Byte Recovery Ratio ($\text{Ratio}_{\text{bytes}}$)
The proportion of original evidence bytes correctly recovered into valid files.
$$\text{Ratio}_{\text{bytes}} = \frac{\sum_{F \in \text{Recovered}} \text{Bytes}_{\text{correct}}(F)}{\sum_{F \in \text{GT}} \text{Size}_{\text{bytes}}(F)}$$

### 3.4. False File Merge Rate ($\text{Rate}_{\text{merge}}$)
Critical forensic error: assembling fragments from two or more distinct source files into a single reconstructed file.
$$\text{Rate}_{\text{merge}} = \frac{\sum_{F \in \text{Recon}} \mathbb{I}(|\{ \text{SourceFile}(f) : f \in F \}| > 1)}{|\text{Recon}|}$$
- **Forensic Impact**: False merges cross chain-of-custody boundaries and can lead to false evidence attribution in court.

### 3.5. False File Split Rate ($\text{Rate}_{\text{split}}$)
Measures original files whose fragments were broken across multiple distinct reconstructed files.
$$\text{Rate}_{\text{split}} = \frac{\sum_{F \in \text{GT}} \mathbb{I}(|\{ \text{ReconFile}(f) : f \in F \}| > 1)}{|\text{GT}|}$$

### 3.6. False Reconstruction Rate ($\text{Rate}_{\text{false}}$)
The proportion of reconstructed files that contain 0 valid fragments or do not correspond to any ground-truth file (e.g. fabricated sequences or chimeric artifacts).
$$\text{Rate}_{\text{false}} = \frac{\sum_{F \in \text{Recon}} \mathbb{I}(|\text{GT\_Members}(F)| == 0)}{|\text{Recon}|}$$

---

## 4. Robustness Metrics

1. **Corruption Tolerance ($\Delta \text{AUC} / \Delta c$)**:
   $$\text{Tolerance}_{\text{corr}} = \frac{\text{Rate}_{\text{exact}}(c=0.0) - \text{Rate}_{\text{exact}}(c=0.20)}{0.20}$$
   Measures degradation slope as payload noise increases from 0% to 20%.

2. **Missing-Fragment Tolerance ($\Delta \text{AUC} / \Delta m$)**:
   Measures partial sequence recovery degradation as block drop rate increases from 0% to 20%.

3. **Interleaving Resistance**:
   $$\text{Resistance}_{\text{interleave}} = \frac{\text{Rate}_{\text{exact}}(10\text{-files})}{\text{Rate}_{\text{exact}}(2\text{-files})}$$

4. **Decoy Rejection Rate**:
   $$\text{Rejection}_{\text{decoy}} = 1.0 - \frac{\text{Decoys Assigned to Reconstructed Files}}{\text{Total Decoys Injected}}$$

---

## 5. Efficiency Metrics

- **Total Execution Runtime ($T_{\text{total}}$)**: Wall-clock time in seconds for complete disk carve and reconstruction.
- **Peak Memory Consumption ($M_{\text{peak}}$)**: Maximum resident set size (RSS) in megabytes during candidate graph construction and beam search.
- **Candidate Graph Cardinality ($|V|, |E|$)**: Number of fragment nodes and scored candidate edges.
- **Pairwise Evaluation Count ($N_{\text{eval}}$)**: Number of inference calls made to the pairwise feature engine and model.
- **Search Time ($T_{\text{search}}$)**: Milliseconds consumed by graph traversal and constraint solvers per file.
