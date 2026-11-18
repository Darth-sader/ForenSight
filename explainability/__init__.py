"""Explainability package (doc §37)."""
from explainability.feature_importance import (grouped_importance,
                                               permutation_importance_manual,
                                               shap_values, top_shap_drivers,
                                               tree_feature_importance)
from explainability.shap_analysis import explain_pair_decision, global_shap_summary

__all__ = ["permutation_importance_manual", "grouped_importance",
           "tree_feature_importance", "shap_values", "top_shap_drivers",
           "explain_pair_decision", "global_shap_summary"]
