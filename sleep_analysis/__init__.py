"""
Sleep Analysis Package

A comprehensive toolkit for analyzing sleep data, including preprocessing,
feature engineering, clustering, causal inference, and visualization.
"""

# Core modules
from . import preprocessing
from . import features
from . import clustering
from . import models
from . import causal
from . import visualization
from . import stats

# Key functions for easier access
from .preprocessing import (
    unpack_episodes,
    calculate_patient_chronotype,
    clean_hr,
    extract_hr_nadir_features,
)
from .features import slope_fixed_len, tda_vector, equalize_hr
from .clustering import embed_and_cluster, meta_cluster_by_centroid
from .models import multinomial_mixed_model
from .visualization import plot_cluster_hr, plot_umap
from .causal import identify_confounders, create_sleep_causal_graph
from .stats import perform_hr_statistics, calculate_patient_cluster_consistency

__version__ = "0.2.0"

__all__ = [
    "preprocessing",
    "features",
    "clustering",
    "models",
    "causal",
    "visualization",
    "stats",
    "unpack_episodes",
    "calculate_patient_chronotype",
    "clean_hr",
    "extract_hr_nadir_features",
    "slope_fixed_len",
    "tda_vector",
    "equalize_hr",
    "embed_and_cluster",
    "meta_cluster_by_centroid",
    "multinomial_mixed_model",
    "plot_cluster_hr",
    "plot_umap",
    "identify_confounders",
    "create_sleep_causal_graph",
    "perform_hr_statistics",
    "calculate_patient_cluster_consistency",
]
