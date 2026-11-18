"""Visualisation package (doc §37)."""
from visualization.embeddings import plot_cluster_composition, plot_embeddings
from visualization.graphs import graph_to_frontend, plot_candidate_path, plot_relationship_graph
from visualization.performance import (plot_calibration, plot_candidate_confidence,
                                       plot_confusion_matrices, plot_feature_importance,
                                       plot_pr_curves, plot_roc_curves, plot_shap_summary,
                                       plot_sweep)

__all__ = ["plot_embeddings", "plot_cluster_composition", "plot_relationship_graph",
           "plot_candidate_path", "graph_to_frontend", "plot_roc_curves",
           "plot_pr_curves", "plot_confusion_matrices", "plot_calibration",
           "plot_feature_importance", "plot_shap_summary", "plot_sweep",
           "plot_candidate_confidence"]
