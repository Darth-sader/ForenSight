"""Evaluation package (doc §26-35)."""
from evaluation.ablation import run_ablation, run_representation_comparison
from evaluation.benchmarks import SignatureCarver, benchmark_components
from evaluation.cross_validation import group_kfold_by_file, no_leakage_check, stratified_kfold
from evaluation.experiments import ExperimentRunner, load_placements, true_sequences_for
from evaluation.metrics import (anomaly_metrics, classification_metrics,
                                pr_curve_points, ranking_metrics,
                                reconstruction_metrics, roc_curve_points, topk_recovery)

__all__ = ["classification_metrics", "reconstruction_metrics", "topk_recovery",
           "ranking_metrics", "anomaly_metrics", "roc_curve_points", "pr_curve_points",
           "stratified_kfold", "group_kfold_by_file", "no_leakage_check",
           "run_ablation", "run_representation_comparison",
           "SignatureCarver", "benchmark_components",
           "ExperimentRunner", "load_placements", "true_sequences_for"]
