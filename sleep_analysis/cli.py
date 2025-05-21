"""
Command Line Interface for Sleep Analysis

This module provides the command-line interface to run the sleep analysis pipeline.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import sem  # Added import for sem
from typer import Option
from tqdm import tqdm
import umap

from sleep_analysis.preprocessing import (
    unpack_episodes,
    calculate_patient_chronotype,
    clean_hr,
    extract_hr_nadir_features,
)
from sleep_analysis.causal import render_save_causal_dag
from sleep_analysis.clustering import (
    calculate_patient_cluster_consistency,
    embed_and_cluster,
)
from sleep_analysis.stats import (
    perform_hr_statistics,
)

from sleep_analysis.features import (
    slope_fixed_len,
    tda_vector,
    equalize_hr,
)
from sleep_analysis.visualization import (
    visualize_consistency_hr_profiles,
    plot_cluster_hr,
    plot_umap,
    plot_hr_summary_by_cluster,
    analyze_cluster_characteristics,
    compare_cluster_characteristics_inferential,
    plot_fudolig_comparison,
    _format_p_value_for_title,  # Import the helper function
)

from sleep_analysis.models import (
    multinomial_mixed_model,
    multinomial_frequentist_model,
    visualize_model_results,
    visualize_posterior_distributions,
)


def get_default_params():
    """Define default parameters for the analysis."""
    params = {
        # Input/output parameters
        "data_path": "/Users/federicostarace/thesis/data/exams_final.pkl",
        "out_dir": "results/",
        # Algorithm parameters
        "equalize_method": "resample",  # Default, but will be overridden in grid search
        "use_causal": True,
        "causal_exposure": None,
        "prior_type": "student_t",
        "prior_scale": 2.0,
        "compare_fudolig": True,
        "cv_method": "loo",  # Use LOO-CV by default
        "min_clusters": 2,  # Minimum number of clusters for silhouette analysis
        "max_clusters": 10,  # Maximum number of clusters for silhouette analysis
        "favor_higher_k": True,  # Apply a small penalty to favor more clusters
        "penalty_weight": 0.05,  # Penalty strength for lower k values
        "reduced_mcmc": False,  # Use reduced MCMC settings to save memory
        # Grid search parameter values
        "eps_values": [0.4, 0.3, 0.2, 0.1],
        "min_samples_values": [5, 10, 15, 20],
        "nadir_weight_values": [1.0],
        "equalize_method_values": ["resample", "paa"],  # HR equalization methods
        "clustering_algorithm_values": [
            # Fixed k=3 versions of all algorithms
            "tda_nadir_umap_kmeans_3",  # K-means with fixed k=3 on UMAP embedding
            # "gmm_slope_3",  # GMM on HR slope features with fixed k=3
            # "hac_dtw_slope_3",  # Hierarchical Agglomerative Clustering with DTW on HR slope with fixed k=3
            # "spectral_clustering_slope_3",  # Spectral Clustering on HR slope features with fixed k=3
            # "gmm_paa_hr_3",  # GMM with fixed k=3 on PAA-HR features
            # "hac_dtw_3",  # Hierarchical Agglomerative Clustering with DTW and fixed k=3
            # "spectral_clustering_paa_hr_3",  # Spectral Clustering with fixed k=3 on PAA-HR features
            # "tda_nadir_umap_kmeans_silhouette_k",  # K-means with optimal k on UMAP embedding
            # "gmm_paa_hr_bic",  # GMM on PAA-HR features with BIC
            # "hac_dtw",  # Hierarchical Agglomerative Clustering with DTW
            # "spectral_clustering_paa_hr",  # Spectral Clustering on PAA-HR features
            # "gmm_slope_bic",  # GMM on HR slope features with BIC (uses first derivative)
            # "hac_dtw_slope",  # Hierarchical Agglomerative Clustering with DTW on HR slope (uses first derivative)
            # "spectral_clustering_slope",  # Spectral Clustering on HR slope features (uses first derivative)
        ],  # Clustering algorithm options
    }
    return params


def run_pipeline(
    df_raw: pd.DataFrame,
    out_dir: Path,
    equalize_method: str = "resample",
    eps: float = 0.4,
    min_samples: int = 10,
    nadir_weight: float = 1.0,
    use_causal: bool = False,
    causal_exposure: Optional[str] = None,
    prior_type: str = "student_t",
    prior_scale: float = 2.0,
    compare_fudolig: bool = False,
    clustering_algorithm: str = "tda_nadir_umap_kmeans_silhouette_k",
    cv_method: Optional[str] = None,
    min_clusters: int = 2,
    max_clusters: int = 10,
    favor_higher_k: bool = True,
    penalty_weight: float = 0.05,
    reduced_mcmc: bool = False,
) -> dict:
    """
    Run the sleep analysis pipeline with the specified parameters.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw input data
    out_dir : Path
        Output directory
    equalize_method : str
        Method for equalizing HR length
    eps : float
        DBSCAN epsilon parameter (deprecated, kept for backward compatibility)
    min_samples : int
        DBSCAN min_samples parameter (deprecated, kept for backward compatibility)
    nadir_weight : float
        Weight for nadir timing in clustering
    use_causal : bool
        Whether to use causal adjustment
    causal_exposure : str or None
        Variable to use as exposure
    prior_type : str
        Prior distribution type
    prior_scale : float
        Scale for prior distributions
    compare_fudolig : bool
        Whether to compare with Fudolig approach (deprecated)
    clustering_algorithm : str
        Clustering algorithm to use:
        - "tda_nadir_umap_kmeans_silhouette_k": K-means clustering with optimal k on UMAP embedding
        - "gmm_paa_hr_bic": GMM on PAA-HR features with BIC
        - "hac_dtw": Hierarchical Agglomerative Clustering with DTW
        - "spectral_clustering_paa_hr": Spectral Clustering on PAA-HR features
        - "gmm_slope_bic": GMM on HR slope features with BIC (uses first derivative)
        - "hac_dtw_slope": Hierarchical Agglomerative Clustering with DTW on HR slope (uses first derivative)
        - "spectral_clustering_slope": Spectral Clustering on HR slope features (uses first derivative)
        - "kmeans_fixed_k3": K-means with fixed k=3 on UMAP embedding
        - "gmm_fixed_k3": GMM with fixed k=3 on original features
        - "spectral_fixed_k3": Spectral Clustering with fixed k=3
        - "hac_dtw_fixed_k3": Hierarchical Agglomerative Clustering with DTW and fixed k=3
        - "gmm_slope_fixed_k3": GMM on HR slope features with fixed k=3
    cv_method : str or None
        Cross-validation method to use ("loo" or "waic"), None for no CV
    min_clusters : int
        Minimum number of clusters for silhouette analysis
    max_clusters : int
        Maximum number of clusters for silhouette analysis
    favor_higher_k : bool
        Whether to apply a small penalty to favor higher cluster counts
    penalty_weight : float
        Strength of the penalty for lower k values
    reduced_mcmc : bool, default=False
        Whether to use reduced MCMC settings to save memory

    Returns
    -------
    dict
        Statistics about the clustering results
    """
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save parameter settings to a text file
    with open(out_dir / "parameters.txt", "w") as f:
        f.write(f"equalize_method: {equalize_method}\n")
        f.write(f"eps: {eps}\n")
        f.write(f"min_samples: {min_samples}\n")
        f.write(f"nadir_weight: {nadir_weight}\n")
        f.write(f"use_causal: {use_causal}\n")
        f.write(f"causal_exposure: {causal_exposure}\n")
        f.write(f"prior_type: {prior_type}\n")
        f.write(f"prior_scale: {prior_scale}\n")
        f.write(f"compare_fudolig: {compare_fudolig}\n")
        f.write(f"clustering_algorithm: {clustering_algorithm}\n")
        f.write(f"cv_method: {cv_method}\n")
        f.write(f"min_clusters: {min_clusters}\n")
        f.write(f"max_clusters: {max_clusters}\n")
        f.write(f"favor_higher_k: {favor_higher_k}\n")
        f.write(f"penalty_weight: {penalty_weight}\n")
        f.write(f"reduced_mcmc: {reduced_mcmc}\n")

    # Memory monitoring
    try:
        import psutil

        memory_usage_start = psutil.Process().memory_info().rss / 1024**2  # MB
        print(f"Memory usage at start: {memory_usage_start:.1f} MB")
    except ImportError:
        print("psutil not available for memory monitoring")
        memory_usage_start = None

    # Generate and save the causal DAG if using causal inference
    if use_causal:
        print("\nGenerating causal DAG visualization...")
        dag_path = render_save_causal_dag(
            output_dir=out_dir, filename="causal_dag", format="png", view=False
        )
        if dag_path:
            print(f"Causal DAG saved to: {dag_path}")

    # Run the pipeline
    print(f"\n{'='*80}")
    print(
        f"Running pipeline with: eps={eps}, min_samples={min_samples}, nadir_weight={nadir_weight}, "
        f"clustering_algorithm={clustering_algorithm}"
    )
    print(f"{'='*80}")

    # Unpack episodes
    eps_df = unpack_episodes(df_raw)
    print(f"Unpacked {len(eps_df):,} episodes from {len(df_raw):,} patients.")

    # Calculate patient-level chronotype
    eps_df = calculate_patient_chronotype(eps_df)

    # Build features including nadir timing
    processed_features = {
        "slope": [],
        "tda_vec": [],
        "episode_hr_mean": [],
        "hr_eq": [],
        "nadir_time_pct": [],
        "nadir_time_hours": [],
        "nadir_hr": [],
        "time_to_nadir_normalized": [],
        "keep": [],
    }

    # Add tqdm progress bar
    for idx, row in tqdm(
        eps_df.iterrows(),
        desc="Processing heart rate data",
        unit="episode",
        total=len(eps_df),
    ):
        hr_arr = row["hr_vector"]
        duration_sec = row["duration_secs"]

        hr = clean_hr(hr_arr)
        if len(hr) < 30:  # Using WINDOW_SIZE from features module
            processed_features["keep"].append(False)
            processed_features["slope"].append(
                np.zeros(100, np.float32)
            )  # TARGET_LEN from features
            processed_features["tda_vec"].append(np.zeros(2, np.float32))
            processed_features["episode_hr_mean"].append(np.nan)
            processed_features["hr_eq"].append(np.zeros(100, np.float32))  # TARGET_LEN
            processed_features["nadir_time_pct"].append(np.nan)
            processed_features["nadir_time_hours"].append(np.nan)
            processed_features["nadir_hr"].append(np.nan)
            processed_features["time_to_nadir_normalized"].append(np.nan)
            continue

        # Calculate mean heart rate for this episode
        episode_hr_mean = np.nanmean(hr)

        # Get equalized HR for visualization
        hr_eq = equalize_hr(hr, method=equalize_method)

        # Extract nadir timing features - get actual hr_times and hr_values
        actual_hr_times = row.get("hr_times")
        actual_hr_values = row.get("hr_values")

        # Check if we have valid actual timing data
        has_valid_actual_timing = (
            actual_hr_times is not None
            and actual_hr_values is not None
            and len(actual_hr_times) > 0
            and len(actual_hr_values) > 0
            and len(actual_hr_times) == len(actual_hr_values)
        )

        # Use actual timing data if available, otherwise use equalized HR
        if has_valid_actual_timing:
            # We've already checked that these are not None in has_valid_actual_timing
            # Add explicit check to avoid type errors
            if actual_hr_times is not None and actual_hr_values is not None:
                hr_times_list = list(actual_hr_times)
                hr_values_list = list(actual_hr_values)

                nadir_features = extract_hr_nadir_features(
                    hr, duration_sec, hr_times=hr_times_list, hr_values=hr_values_list
                )
            else:
                # Fallback to synthetic time points
                synthetic_times = np.linspace(0, duration_sec, len(hr_eq))
                nadir_features = extract_hr_nadir_features(
                    hr_eq,
                    duration_sec,
                    hr_times=synthetic_times.tolist(),
                    hr_values=hr_eq.tolist(),
                )
        else:
            # If no raw precise times, use hr_eq for consistent nadir calculation
            # Create synthetic time points evenly distributed over the sleep episode
            synthetic_times = np.linspace(0, duration_sec, len(hr_eq))
            nadir_features = extract_hr_nadir_features(
                hr_eq,
                duration_sec,
                hr_times=synthetic_times.tolist(),
                hr_values=hr_eq.tolist(),
            )

        # Store all features
        processed_features["keep"].append(True)
        processed_features["episode_hr_mean"].append(episode_hr_mean)
        processed_features["hr_eq"].append(hr_eq)  # Store equalized HR
        slope = slope_fixed_len(hr, method=equalize_method)
        processed_features["slope"].append(slope)
        processed_features["tda_vec"].append(tda_vector(slope))
        processed_features["nadir_time_pct"].append(nadir_features["nadir_time_pct"])
        processed_features["nadir_time_hours"].append(
            nadir_features["nadir_time_hours"]
        )
        processed_features["nadir_hr"].append(nadir_features["nadir_hr"])
        processed_features["time_to_nadir_normalized"].append(
            nadir_features["time_to_nadir_normalized"]
        )

    # Assign with .loc to avoid SettingWithCopyWarning
    for key, value_list in processed_features.items():
        eps_df.loc[:, key] = value_list

    # Create a clean copy of the DataFrame after filtering
    eps_df = eps_df.query("keep").copy()
    X = np.vstack(list(eps_df["tda_vec"].values))

    # Get nadir times for clustering if available
    nadir_times = None
    if "nadir_time_pct" in eps_df.columns and nadir_weight > 0:
        # Convert to numpy array explicitly to avoid type issues
        valid_nadirs = ~pd.isna(eps_df["nadir_time_pct"].values)
        if np.any(valid_nadirs):
            nadir_times = np.array(eps_df["nadir_time_pct"].values, dtype=np.float32)
            print(f"Using nadir timing in clustering with weight {nadir_weight}")
        else:
            print(
                "Nadir timing available but all values are NaN. Not using in clustering."
            )
    else:
        print("Not using nadir timing in clustering")

    # Prepare feature matrix for UMAP, potentially including nadir timing
    X_combined = X
    if nadir_times is not None:
        # Handle NaN values in nadir times
        valid_nadir = ~np.isnan(nadir_times)
        if np.any(valid_nadir):
            # Fill NaN values with median
            nadir_clean = nadir_times.copy()
            nadir_median = np.nanmedian(nadir_times)
            nadir_clean[~valid_nadir] = nadir_median

            # Normalize nadir times to [0,1] range if we have variation
            nadir_min = np.nanmin(nadir_times)
            nadir_max = np.nanmax(nadir_times)
            nadir_range = nadir_max - nadir_min

            if nadir_range > 0:
                # Ensure nadir_range is a float for division to satisfy linter
                nadir_norm = (nadir_clean - nadir_min) / float(nadir_range)
            else:
                # Handle the case where all values are identical or all NaNs resulted in range 0
                nadir_norm = np.zeros_like(nadir_clean)

            # Weight and combine with other features
            X_combined = np.column_stack([X, nadir_norm * nadir_weight])
            print(
                f"Combined feature matrix shape with nadir timing: {X_combined.shape}"
            )

    # Create and fit UMAP
    print("\nGenerating UMAP embedding...")
    n_neighbors = min(
        30, max(2, len(X_combined) - 1)
    )  # Ensure n_neighbors is valid (at least 2)
    umap_reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=0.1, metric="euclidean", random_state=42
    )

    # Get the 2D embedding
    embedding_result = umap_reducer.fit_transform(X_combined)
    umap_embedding = np.asarray(embedding_result, dtype=np.float32)

    # Generate k-distance plot to help choose eps parameter
    if len(umap_embedding) > min_samples:
        print("\nGenerating k-distance plot to help choose DBSCAN eps parameter...")
        from sklearn.neighbors import NearestNeighbors

        # Use min_samples for k-distance plot
        k_for_dist_plot = min_samples

        nbrs = NearestNeighbors(n_neighbors=k_for_dist_plot).fit(umap_embedding)
        distances, indices = nbrs.kneighbors(umap_embedding)
        # Sort the distances to the k-th nearest neighbor
        k_distances = np.sort(distances[:, -1])

        # Create the k-distance plot
        plt.figure(figsize=(10, 6))
        plt.plot(k_distances)
        plt.axhline(y=eps, color="r", linestyle="--", label=f"Current eps = {eps}")
        plt.xlabel("Points (sorted by distance)")
        plt.ylabel(f"Distance to {k_for_dist_plot}-th nearest neighbor")
        plt.title("k-distance Plot for DBSCAN eps Parameter Selection")
        plt.legend()

        # Add some guidelines for potential eps values at visible "knees" in the curve
        from sklearn.neighbors import KernelDensity
        from scipy.signal import find_peaks

        # Calculate approximate derivative
        gradient = np.gradient(k_distances)

        # Use KDE to smooth the gradient
        try:
            # Reshape for KDE
            X_grad = gradient.reshape(-1, 1)
            # Fit KDE to smooth the gradient
            kde = KernelDensity(kernel="gaussian", bandwidth=0.05).fit(X_grad)
            # Sample points
            X_plot = np.linspace(min(gradient), max(gradient), 1000).reshape(-1, 1)
            # Get log density
            log_dens = kde.score_samples(X_plot)

            # Find peaks in the negative derivative (inflection points in original curve)
            peaks, _ = find_peaks(-np.exp(log_dens))

            # Get original indices
            if len(peaks) > 0:
                # Only use a few peaks (first 3-5)
                max_peaks = min(5, len(peaks))
                selected_peaks = peaks[:max_peaks]

                for i, peak_idx in enumerate(selected_peaks):
                    # Get corresponding position in original k_distances
                    peak_pos = int(peak_idx / len(X_plot) * len(k_distances))
                    if peak_pos < len(k_distances):
                        eps_suggestion = k_distances[peak_pos]
                        if eps_suggestion > 0:  # Avoid very small values
                            plt.axhline(
                                y=eps_suggestion,
                                color=f"C{i+1}",
                                linestyle=":",
                                label=f"Suggested eps {i+1} = {eps_suggestion:.3f}",
                            )
        except Exception as e:
            print(f"Could not generate eps suggestions: {e}")

        plt.legend()
        dist_plot_path = out_dir / "k_distance_plot.png"
        plt.savefig(dist_plot_path, dpi=300)
        plt.close()
        print(f"k-distance plot saved to {dist_plot_path}")

    # Embedding & clustering based on selected algorithm
    print(f"\nPerforming clustering using {clustering_algorithm} algorithm...")
    emb, clusters, k2_labels = embed_and_cluster(
        X=X,
        eps=eps,
        min_samples=min_samples,
        nadir_times=nadir_times,
        nadir_weight=nadir_weight,
        compare_fudolig=compare_fudolig and clustering_algorithm == "umap_dbscan",
        algorithm=clustering_algorithm,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
        favor_higher_k=favor_higher_k,
        penalty_weight=penalty_weight,
    )

    # Add cluster and embedding columns
    eps_df["cluster"] = (
        clusters
        if len(clusters) == len(eps_df)
        else np.zeros(len(eps_df), dtype=np.int32)
    )

    if (
        k2_labels is not None
        and len(k2_labels) == len(eps_df)
        and clustering_algorithm == "umap_dbscan"
    ):
        eps_df["fudolig_cluster"] = k2_labels
        print("Added Fudolig-style k=2 clustering for comparison")

    # Add UMAP coordinates
    if len(emb) == len(eps_df) and emb.shape[1] >= 2:
        eps_df["umap1"] = emb[:, 0]
        eps_df["umap2"] = emb[:, 1]
    else:
        print(f"Warning: UMAP embedding shape doesn't match data length")
        eps_df["umap1"] = np.zeros(len(eps_df))
        eps_df["umap2"] = np.zeros(len(eps_df))

    # Remove meta-clustering completely
    use_meta = False

    # --- Post-Clustering Analysis (Patient consistency, visualizations, stats) ---
    patient_consistency = calculate_patient_cluster_consistency(eps_df, use_meta=False)
    if not patient_consistency.empty:
        patient_consistency.to_csv(
            out_dir / "patient_cluster_consistency.csv", index=False
        )

    # Prepare HR data by cluster for visualization
    hr_by_cluster = {}
    slope_by_cluster = {}  # Add initialization here to avoid unbound variable error

    for cluster_id, group_df in eps_df.groupby("cluster"):
        if cluster_id == -1:  # Skip noise points
            continue
        hr_by_cluster[cluster_id] = np.stack(list(group_df["hr_eq"].values))

    # For slope-based methods, prepare slope data by cluster too
    is_slope_based = clustering_algorithm in [
        "gmm_slope_bic",
        "hac_dtw_slope",
        "spectral_clustering_slope",
    ]
    if is_slope_based:
        for cluster_id, group_df in eps_df.groupby("cluster"):
            if cluster_id == -1:  # Skip noise points
                continue
            # Calculate slopes for all HR series in the cluster
            slopes = []
            for hr_eq in group_df["hr_eq"].values:
                slope = slope_fixed_len(hr_eq, method="resample")
                slopes.append(slope)
            slope_by_cluster[cluster_id] = np.stack(slopes)

    # Create plots
    print("\nGenerating visualizations...")

    # Plot UMAP embedding with cluster labels
    # Note: UMAP embedding is always calculated on original features for consistent visualization
    if (
        "umap1" in eps_df.columns
        and "umap2" in eps_df.columns
        and not eps_df[["umap1", "umap2"]].isnull().all().all()
    ):
        fig_umap = plot_umap(
            df=eps_df.rename(columns={"umap1": "umap1", "umap2": "umap2"}),
            out_path=out_dir / "embedding_plot.png",
        )

        # Add title separately as a text annotation if needed
        plt.figure()
        plt.text(
            0.5,
            0.5,
            f"UMAP Embedding with {clustering_algorithm} Clustering",
            ha="center",
            va="center",
            fontsize=12,
        )
        plt.axis("off")
        plt.savefig(out_dir / "embedding_title.png")
        plt.close()

    # Plot HR patterns by cluster
    if hr_by_cluster:
        hr_title = f"Heart Rate Patterns by Cluster ({clustering_algorithm})"
        fig_hr = plot_cluster_hr(
            hr_data_by_cluster=hr_by_cluster,
            out_path=out_dir / "hr_by_cluster.png",
        )

        # For slope-based methods, also plot slope patterns
        if is_slope_based:
            # Check if the dictionary has content directly
            if slope_by_cluster and any(
                len(slopes) > 0 for slopes in slope_by_cluster.values()
            ):
                fig_slope = plot_cluster_hr(
                    hr_data_by_cluster=slope_by_cluster,
                    out_path=out_dir / "slope_by_cluster.png",
                )

                # Add custom title/ylabel with matplotlib
                plt.figure()
                plt.text(
                    0.5,
                    0.5,
                    f"Heart Rate Slope (First Derivative) Patterns by Cluster ({clustering_algorithm})",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                plt.axis("off")
                plt.savefig(out_dir / "slope_title.png")
                plt.close()

    # Plot HR summary statistics by cluster
    if hr_by_cluster and any(len(hr_array) > 0 for hr_array in hr_by_cluster.values()):
        try:
            hr_stats_df = pd.DataFrame(
                [
                    {
                        "cluster": cluster_id,
                        "mean_hr": np.mean(np.mean(hr_by_cluster[cluster_id], axis=1)),
                        "min_hr": np.mean(
                            [curve.min() for curve in hr_by_cluster[cluster_id]]
                        ),
                        "max_hr": np.mean(
                            [curve.max() for curve in hr_by_cluster[cluster_id]]
                        ),
                    }
                    for cluster_id in hr_by_cluster.keys()
                ]
            )

            fig_hr_stats = plot_hr_summary_by_cluster(
                stats_df=hr_stats_df,
                out_path=out_dir / "hr_stats_by_cluster.png",
            )
        except Exception as e:
            print(f"Error generating HR summary statistics: {e}")
            print("Skipping HR summary visualization")
    else:
        print(
            "Warning: No valid HR data available for summary statistics. Skipping visualization."
        )
        hr_stats_df = pd.DataFrame(columns=["cluster", "mean_hr", "min_hr", "max_hr"])

    # Create a comprehensive visualization comparing clusters
    plt.figure(figsize=(16, 12))

    # Plot 1: UMAP Embedding with Clusters
    plt.subplot(2, 2, 1)
    scatter = plt.scatter(
        eps_df["umap1"],
        eps_df["umap2"],
        c=eps_df["cluster"],
        cmap="viridis",
        alpha=0.8,
        s=50,
    )
    plt.colorbar(scatter, label="Cluster")
    plt.title(f"UMAP Embedding with {clustering_algorithm}")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")

    # Plot 2: Mean HR Pattern by Cluster (SEM)
    plt.subplot(2, 2, 2)
    for cluster_id, hr_data in hr_by_cluster.items():
        mean_hr = np.mean(hr_data, axis=0)
        sem_hr = sem(hr_data, axis=0)  # New: using sem
        x = np.arange(len(mean_hr))
        plt.plot(x, mean_hr, label=f"Cluster {cluster_id}")
        plt.fill_between(
            x, mean_hr - sem_hr, mean_hr + sem_hr, alpha=0.2
        )  # New: using sem
    plt.title("Mean HR Pattern by Cluster (SEM)")  # Updated title
    plt.xlabel("Normalized Time")
    plt.ylabel("Heart Rate (bpm)")
    plt.legend()

    # Plot 3: Box Plot of HR Statistics by Cluster
    plt.subplot(2, 2, 3)
    try:
        if "episode_hr_mean" in eps_df.columns:
            sns.boxplot(x="cluster", y="episode_hr_mean", data=eps_df)
            plt.title("Mean HR Distribution by Cluster")
            plt.xlabel("Cluster")
            plt.ylabel("Mean Heart Rate (bpm)")
        else:
            plt.text(0.5, 0.5, "No HR mean data available", ha="center", va="center")
            plt.title("Mean HR Distribution (No Data)")
    except Exception as e:
        print(f"Error plotting HR distribution: {e}")
        plt.text(
            0.5, 0.5, f"Error plotting HR data: {str(e)}", ha="center", va="center"
        )
        plt.title("Mean HR Distribution (Error)")

    # Plot 4: Patient Consistency by Cluster
    plt.subplot(2, 2, 4)
    if not patient_consistency.empty:
        sns.barplot(
            x="modal_cluster",
            y="cluster_consistency",
            data=patient_consistency,
            ci=None,
        )
        plt.title("Patient Consistency by Modal Cluster")
        plt.xlabel("Modal Cluster")
        plt.ylabel("Consistency Score (higher = more consistent)")
    else:
        plt.text(
            0.5,
            0.5,
            "Insufficient data for consistency analysis",
            ha="center",
            va="center",
        )
        plt.title("Patient Consistency (No Data)")

    plt.tight_layout()
    plt.savefig(out_dir / "comprehensive_cluster_analysis.png", dpi=300)
    plt.close()

    # Analyze cluster characteristics
    print("\nAnalyzing cluster characteristics...")
    analyze_cluster_characteristics(df=eps_df, out_dir=out_dir)

    # Inferential statistics
    print("\nPerforming inferential statistical tests for cluster comparisons...")
    compare_cluster_characteristics_inferential(eps_df, out_dir=out_dir)

    # NEW: Advanced inferential statistics on clusters
    print("\nPerforming advanced inferential analysis on clusters...")
    cluster_infer_results = analyze_clusters_inferential(eps_df, out_dir)

    # NEW: Analyze HR metrics, chronotype relationships, and sleep indices
    print(
        "\nAnalyzing relationships between HR metrics, chronotype, and sleep indices..."
    )
    hr_chrono_results = analyze_hr_chronotype_relationships(eps_df, out_dir)

    # Fudolig comparison if requested and using umap_dbscan
    if (
        compare_fudolig
        and "fudolig_cluster" in eps_df.columns
        and clustering_algorithm == "umap_dbscan"
    ):
        print("\nCreating comparison with Fudolig et al. approach...")
        plot_fudolig_comparison(eps_df, out_path=out_dir / "fudolig_comparison.png")

    # Add save_csv to help debug cross-validation issues
    # This is now set to False by default to avoid creating very large files.
    # Set to True if detailed MCMC samples are needed for debugging.
    save_csv = False

    # Check memory usage before multinomial model
    try:
        if memory_usage_start is not None:
            import psutil

            current_memory = psutil.Process().memory_info().rss / 1024**2  # MB
            print(f"Memory usage before models: {current_memory:.1f} MB")
            print(f"Increase: {current_memory - memory_usage_start:.1f} MB")

            # Auto-adjust reduced_mcmc if memory usage is high
            if current_memory > 8000 and not reduced_mcmc:  # 8GB threshold
                print(
                    "High memory usage detected. Automatically enabling reduced_mcmc."
                )
                reduced_mcmc = True
    except Exception as e:
        print(f"Error monitoring memory: {e}")

    # Define all target relationships to test
    target_relationships = [
        ("patient_chronotype", "patient_chronotype -> cluster", "chronotype"),
        ("age", "age -> cluster", "age"),
        ("abs_chronotype_desync", "abs_chronotype_desync -> cluster", "abs_desync"),
        ("chronotype_desync", "chronotype_desync -> cluster", "desync"),
        ("se", "se -> cluster", "se"),
        ("sex", "sex -> cluster", "sex"),
        ("sri", "sri -> cluster", "sri"),
        ("waso", "waso -> cluster", "waso"),
    ]

    # Run models for each target relationship
    print("\nFitting models for multiple target relationships...")

    # Create a models_outputs subdirectory for all model outputs
    models_output_dir = out_dir / "models_outputs"
    models_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model outputs will be saved to: {models_output_dir}")

    for var_name, relationship, file_suffix in target_relationships:
        # Skip if the variable isn't in the dataset
        if var_name not in eps_df.columns:
            print(f"Skipping {relationship} - {var_name} not in dataset")
            continue

        # Create a target-specific directory
        target_dir = relationship.replace(" -> ", "_to_").replace(" ", "_")
        relationship_dir = models_output_dir / target_dir
        relationship_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nFitting model for target relationship: {relationship}")
        output_path = relationship_dir / f"multinomial_model_{file_suffix}_summary.txt"

        # Run the model
        multinomial_mixed_model(
            eps_df,
            out_path=output_path,
            use_meta=False,
            use_causal=use_causal,
            target_relationship=relationship,
            prior_scale=prior_scale,
            reduced_mcmc=reduced_mcmc,
        )

        # Linear mixed model (frequentist approach)
        print("\nFitting linear mixed model (frequentist approach)...")
        multinomial_frequentist_model(
            eps_df,
            out_path=relationship_dir
            / f"freq_multinomial_model_{file_suffix}_summary.txt",
            target_relationship=relationship,
        )

    # Create visualizations of model results
    print("\nGenerating model result visualizations...")
    visualize_model_results(models_output_dir)

    # Generate direct posterior visualizations (more robust method)
    print("\nGenerating direct posterior distribution visualizations...")
    visualize_posterior_distributions(models_output_dir)

    # Final memory usage check
    try:
        if memory_usage_start is not None:
            import psutil

            final_memory = psutil.Process().memory_info().rss / 1024**2  # MB
            print(f"Final memory usage: {final_memory:.1f} MB")
            print(f"Total increase: {final_memory - memory_usage_start:.1f} MB")
    except Exception:
        pass

    # Return statistics about the clustering results
    stats = {
        "n_clusters": len(hr_by_cluster),  # Number of clusters excluding noise
        "noise_points": sum(clusters == -1),
        "cluster_sizes": {i: sum(clusters == i) for i in np.unique(clusters) if i >= 0},
        "total_points": len(clusters),
        "stopped_early": False,
    }

    # Save statistics to file
    with open(out_dir / "clustering_stats.txt", "w") as f:
        f.write(f"Number of clusters: {stats['n_clusters']}\n")
        f.write(
            f"Noise points: {stats['noise_points']} ({stats['noise_points']/stats['total_points']*100:.1f}%)\n"
        )
        f.write("Cluster sizes:\n")
        for cluster_id, size in stats["cluster_sizes"].items():
            f.write(
                f"  Cluster {cluster_id}: {size} ({size/stats['total_points']*100:.1f}%)\n"
            )

        # Add consistency metrics to stats file
        f.write("\nCluster Consistency Metrics:\n")
        if not patient_consistency.empty:
            mean_consistency = patient_consistency["cluster_consistency"].mean()
            high_consistency_percent = (
                patient_consistency["cluster_consistency"] > 0.8
            ).mean() * 100
            mean_modal = patient_consistency["modal_proportion"].mean()

            f.write(f"  Mean Consistency: {mean_consistency:.2f}\n")
            f.write(f"  % Patients >0.8 Consistency: {high_consistency_percent:.1f}\n")
            f.write(f"  Mean Modal Proportion: {mean_modal:.2f}\n")
        else:
            f.write("  No consistency metrics available\n")

    # Save the detailed episode dataframe
    detailed_df_path = out_dir / "processed_episodes_detailed.pkl"
    try:
        # Select relevant columns to save to avoid excessively large files if some columns are huge and not needed for presentation
        # This list can be adjusted based on actual needs for Manim plots
        cols_to_save = [
            "id",
            "episode_id",
            "duration_secs",
            "hr_vector",
            "hr_eq",
            "slope",
            "tda_vec",
            "cluster",
            "umap1",
            "umap2",
            "nadir_time_pct",
            "nadir_hr",
            "age",
            "sex",
            "patient_chronotype",
            "abs_chronotype_desync",
            "sri",
            "se",
            "waso",
            "sfi",
        ]
        # Filter out columns that might not exist to prevent errors
        existing_cols_to_save = [col for col in cols_to_save if col in eps_df.columns]
        eps_df_to_save = eps_df[existing_cols_to_save]

        eps_df_to_save.to_pickle(detailed_df_path)
        print(f"Detailed episode data saved to: {detailed_df_path}")
    except Exception as e:
        print(f"Error saving detailed episode data to {detailed_df_path}: {e}")

    print(f"\nAnalysis complete! Results saved to {out_dir}")
    return stats


def create_grid_search_summary(
    out_dir,
    eps_values,
    min_samples_values,
    nadir_weight_values,
    equalize_method_values,
    clustering_algorithm_values,
    summary_df,
):
    """Create visualizations summarizing grid search results."""
    print("\nCreating grid search summary visualization...")
    try:
        # Calculate percentage of noise points
        summary_df["noise_percent"] = (
            summary_df["noise_points"]
            / (summary_df["noise_points"] + summary_df["valid_points"])
            * 100
        )

        # For each combination of equalize method and clustering algorithm, create separate visualizations
        for eq_method in equalize_method_values:
            for clust_alg in clustering_algorithm_values:
                subset_df = summary_df[
                    (summary_df["equalize_method"] == eq_method)
                    & (summary_df["clustering_algorithm"] == clust_alg)
                ]

                if len(subset_df) == 0:
                    print(
                        f"No data for equalize_method={eq_method}, clustering_algorithm={clust_alg}, skipping visualizations"
                    )
                    continue

                print(
                    f"Creating visualizations for equalize_method={eq_method}, clustering_algorithm={clust_alg}"
                )

                # Create visualizations
                fig, axes = plt.subplots(1, 2, figsize=(16, 7))

                # Plot number of clusters
                for i, nw in enumerate(nadir_weight_values):
                    subset = subset_df[subset_df["nadir_weight"] == nw]

                    if len(subset) == 0:
                        continue

                    pivot_clusters = subset.pivot(
                        index="eps", columns="min_samples", values="n_clusters"
                    )
                    heat = axes[0].imshow(
                        pivot_clusters, aspect="auto", interpolation="nearest"
                    )
                    axes[0].set_title(
                        f"Number of Clusters (nadir_weight={nw}, {eq_method}, {clust_alg})"
                    )
                    axes[0].set_xlabel("min_samples")
                    axes[0].set_ylabel("eps")
                    axes[0].set_xticks(range(len(min_samples_values)))
                    axes[0].set_yticks(range(len(eps_values)))
                    axes[0].set_xticklabels(min_samples_values)
                    axes[0].set_yticklabels(eps_values)
                    plt.colorbar(heat, ax=axes[0])
                    plt.savefig(
                        out_dir
                        / f"grid_clusters_nadir_{nw}_{eq_method}_{clust_alg}.png",
                        dpi=300,
                    )
                    plt.close()

                # Plot percentage of noise points
                for i, nw in enumerate(nadir_weight_values):
                    subset = subset_df[subset_df["nadir_weight"] == nw]

                    if len(subset) == 0:
                        continue

                    pivot_noise = subset.pivot(
                        index="eps", columns="min_samples", values="noise_percent"
                    )
                    heat = axes[1].imshow(
                        pivot_noise, aspect="auto", interpolation="nearest"
                    )
                    axes[1].set_title(
                        f"Noise Points % (nadir_weight={nw}, {eq_method}, {clust_alg})"
                    )
                    axes[1].set_xlabel("min_samples")
                    axes[1].set_ylabel("eps")
                    axes[1].set_xticks(range(len(min_samples_values)))
                    axes[1].set_yticks(range(len(eps_values)))
                    axes[1].set_xticklabels(min_samples_values)
                    axes[1].set_yticklabels(eps_values)
                    plt.colorbar(heat, ax=axes[1])
                    plt.savefig(
                        out_dir / f"grid_noise_nadir_{nw}_{eq_method}_{clust_alg}.png",
                        dpi=300,
                    )
                    plt.close()

        # 3D scatter plot of parameter space - create separate plots for each clustering algorithm
        try:
            from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

            # Create separate 3D plots for each combination of equalize method and clustering algorithm
            for eq_method in equalize_method_values:
                for clust_alg in clustering_algorithm_values:
                    subset_df = summary_df[
                        (summary_df["equalize_method"] == eq_method)
                        & (summary_df["clustering_algorithm"] == clust_alg)
                    ]

                    if len(subset_df) == 0:
                        continue

                    fig = plt.figure(figsize=(10, 8))
                    ax3d = fig.add_subplot(111, projection="3d")
                    scatter = ax3d.scatter(
                        subset_df["eps"],
                        subset_df["min_samples"],
                        subset_df["nadir_weight"],
                        c=subset_df["n_clusters"],
                        alpha=0.7,
                        cmap="viridis",
                    )
                    ax3d.set_xlabel("eps")
                    ax3d.set_ylabel("min_samples")

                    # This is valid for Axes3D - ignore linter warning
                    ax3d.set_zlabel("nadir_weight")  # type: ignore

                    ax3d.set_title(
                        f"Number of Clusters in Parameter Space ({eq_method}, {clust_alg})"
                    )
                    plt.colorbar(scatter, ax=ax3d, label="Number of Clusters")
                    plt.tight_layout()
                    plt.savefig(
                        out_dir / f"grid_search_3d_{eq_method}_{clust_alg}.png", dpi=300
                    )
                    plt.close()
        except Exception as e:
            print(f"Error creating 3D plots: {str(e)}")

    except Exception as e:
        print(f"Error creating summary visualization: {str(e)}")


def visualize_cluster_consistency(
    consistency_df, out_dir, metric_name: str = "cluster_consistency"
):
    """Create visualizations of patient-level cluster consistency or modal proportion metrics.

    Parameters
    ----------
    consistency_df : pd.DataFrame
        DataFrame with patient consistency metrics (must include the specified metric_name)
    out_dir : Path
        Output directory for saving plots
    metric_name : str, default="cluster_consistency"
        The name of the metric column to analyze ("cluster_consistency" or "modal_proportion")
    """
    # Check if we have data and the specified metric column
    if consistency_df.empty or metric_name not in consistency_df.columns:
        print(f"No data or metric '{metric_name}' available for visualization")
        return {}

    # Create a directory for the specific metric visualizations
    metric_dir = out_dir / f"{metric_name}_analysis"
    metric_dir.mkdir(exist_ok=True, parents=True)

    # Plot distribution of the specified metric scores
    plt.figure(figsize=(10, 6))
    sns.histplot(consistency_df[metric_name], kde=True, bins=20)
    plt.title(f"Distribution of Patient {metric_name.replace('_', ' ').title()} Scores")
    plt.xlabel(
        f"{metric_name.replace('_', ' ').title()} (higher = more consistent/modal)"
    )
    plt.ylabel("Count")
    mean_val = consistency_df[metric_name].mean()
    plt.axvline(
        mean_val,
        color="r",
        linestyle="--",
        label=f"Mean: {mean_val:.2f}",
    )
    plt.legend()
    plt.savefig(metric_dir / f"{metric_name}_distribution.png", dpi=300)
    plt.close()

    # Plot relationship between the metric and key variables
    key_vars = ["age", "patient_chronotype", "mean_se", "mean_waso", "sri"]
    available_vars = [var for var in key_vars if var in consistency_df.columns]

    if available_vars:
        n_vars = len(available_vars)
        fig, axes = plt.subplots(n_vars, 1, figsize=(10, 4 * n_vars), squeeze=False)

        for i, var in enumerate(available_vars):
            ax = axes[i, 0]  # Select the correct subplot axis
            sns.scatterplot(x=var, y=metric_name, data=consistency_df, ax=ax)
            ax.set_title(f"{metric_name.replace('_', ' ').title()} vs {var}")
            ax.set_ylabel(metric_name.replace("_", " ").title())  # Set y-axis label

            # Add correlation information if possible
            # Check for sufficient unique non-NaN values in both columns
            valid_mask = (
                consistency_df[var].notna() & consistency_df[metric_name].notna()
            )
            if (
                consistency_df.loc[valid_mask, var].nunique() > 5
                and valid_mask.sum() > 1
            ):
                try:
                    corr, p = stats.spearmanr(
                        consistency_df.loc[valid_mask, var],
                        consistency_df.loc[valid_mask, metric_name],
                    )
                    annotation_text = f"Spearman r: {corr:.2f}, p={p:.3f}"
                except ValueError:
                    annotation_text = "Correlation N/A"
                ax.annotate(
                    annotation_text,
                    xy=(0.05, 0.95),
                    xycoords="axes fraction",
                )
            else:
                # Annotate if correlation couldn't be calculated
                ax.annotate(
                    "Correlation N/A",
                    xy=(0.05, 0.95),
                    xycoords="axes fraction",
                )

        plt.tight_layout()
        plt.savefig(metric_dir / f"{metric_name}_correlations.png", dpi=300)
        plt.close(fig)  # Close the specific figure

    # Create a summary table of consistency metrics - THIS PART REMAINS THE SAME FOR BOTH
    # But we save it within the specific metric directory for context if needed.
    summary = {
        # Calculate summary stats for the specified metric
        f"Mean {metric_name.replace('_', ' ').title()}": consistency_df[
            metric_name
        ].mean(),
        f"Median {metric_name.replace('_', ' ').title()}": consistency_df[
            metric_name
        ].median(),
        f"Min {metric_name.replace('_', ' ').title()}": consistency_df[
            metric_name
        ].min(),
        f"Max {metric_name.replace('_', ' ').title()}": consistency_df[
            metric_name
        ].max(),
        f"Std {metric_name.replace('_', ' ').title()}": consistency_df[
            metric_name
        ].std(),
        # Include other metrics for general context
        "Mean Cluster Consistency": (
            consistency_df["cluster_consistency"].mean()
            if "cluster_consistency" in consistency_df
            else np.nan
        ),
        "Mean Modal Proportion": (
            consistency_df["modal_proportion"].mean()
            if "modal_proportion" in consistency_df
            else np.nan
        ),
        "% Patients >0.8 Consistency": (
            (consistency_df["cluster_consistency"] > 0.8).mean() * 100
            if "cluster_consistency" in consistency_df
            else np.nan
        ),
        "Avg Modal Proportion General": (
            consistency_df["modal_proportion"].mean()
            if "modal_proportion" in consistency_df
            else np.nan
        ),  # Renamed slightly to avoid clash
    }

    # Save summary to text file in the metric-specific directory
    with open(metric_dir / f"{metric_name}_summary.txt", "w") as f:
        f.write(f"Summary statistics focused on {metric_name}:\n")
        for metric, value in summary.items():
            # Use :.2f format specifier, checking for NaN
            f.write(
                f"{metric}: {value:.2f}\n" if pd.notna(value) else f"{metric}: NaN\n"
            )

    # Plot distribution of clusters for most variable patients
    # This part is specifically about consistency (low consistency = variable)
    # We can keep this section tied to 'cluster_consistency' or adapt based on 'metric_name'
    # Let's keep it tied to consistency for now, as 'most variable' is defined by low consistency
    if metric_name == "cluster_consistency" and len(consistency_df) >= 5:
        most_variable = consistency_df.sort_values("cluster_consistency").head(5)
        # ... (rest of the most variable plotting logic remains unchanged) ...
        # Ensure plots are saved in the correct directory
        variable_plots_dir = (
            out_dir / "consistency_analysis"
        )  # Keep this in the main consistency dir
        variable_plots_dir.mkdir(exist_ok=True, parents=True)

        # Save list of most variable patients and their metrics
        with open(variable_plots_dir / "most_variable_patients.txt", "w") as f:
            f.write("Most variable patients (lowest consistency):\n")
            for idx, row in most_variable.iterrows():
                f.write(
                    f"Patient ID: {row['id']}, Consistency: {row['cluster_consistency']:.2f}, "
                )
                f.write(
                    f"Modal Proportion: {row.get('modal_proportion', 'N/A'):.2f}, "
                )  # Use get for safety
                f.write(f"Number of Clusters: {row['n_clusters']}\\n")

        # Create bar plots showing distribution of clusters for each variable patient
        try:
            plt.figure(figsize=(10, 6))
            bar_width = 0.35
            index = np.arange(len(most_variable))

            # Check if 'modal_proportion' exists before plotting
            if "modal_proportion" in most_variable.columns:
                plt.bar(
                    index,
                    most_variable["modal_proportion"],
                    bar_width,
                    label="Modal Cluster Proportion",  # More descriptive label
                )
                plt.bar(
                    index,
                    1 - most_variable["modal_proportion"],
                    bar_width,
                    bottom=most_variable["modal_proportion"],
                    label="Other Clusters Proportion",  # More descriptive label
                )
            else:
                # Fallback if modal_proportion is missing
                plt.text(
                    0.5,
                    0.5,
                    "Modal Proportion data not available",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=plt.gca().transAxes,
                )

            plt.xlabel("Patient ID (Lowest Consistency)")  # Clarify patient selection
            plt.ylabel("Proportion of Episodes")
            plt.title("Cluster Distribution for Most Variable Patients")
            plt.xticks(index, most_variable["id"])
            plt.legend()
            plt.ylim(0, 1)  # Ensure y-axis is 0 to 1

            plt.tight_layout()
            plt.savefig(
                variable_plots_dir / "most_variable_patients_distribution.png", dpi=300
            )  # Slightly clearer name
            plt.close()
        except Exception as e:
            print(
                f"Error creating most variable patients distribution visualization: {e}"
            )

    # Return the summary dictionary
    return summary


def analyze_consistency_inferential(consistency_df, eps_df, out_dir):
    """
    Perform detailed inferential statistical analysis on patient-level cluster consistency.

    Parameters
    ----------
    consistency_df : pd.DataFrame
        DataFrame with patient consistency metrics
    eps_df : pd.DataFrame
        Original episode-level data with clusters
    out_dir : Path
        Output directory for saving results

    Returns
    -------
    dict
        Dictionary with key statistical findings
    """
    if consistency_df.empty:
        print("No consistency data available for inferential analysis")
        return {}

    # Create output directory
    stats_dir = out_dir / "consistency_analysis" / "inferential_stats"
    stats_dir.mkdir(exist_ok=True, parents=True)

    results = {}
    significant_findings = []

    # 1. Multiple regression analysis to identify predictors of consistency
    print("Running multiple regression analysis for consistency predictors...")

    # Predictors to test - both demographic and sleep-related
    potential_predictors = [
        "age",
        "sex",
        "patient_chronotype",
        "sri",
        "mean_se",
        "mean_waso",
        "mean_episode_hr_mean",
        "mean_nadir_time_pct",
        "mean_nadir_hr",
        "mean_duration_secs",
    ]

    # Filter to available predictors
    predictors = [p for p in potential_predictors if p in consistency_df.columns]

    if (
        len(predictors) >= 2
    ):  # Need at least a couple predictors for meaningful regression
        try:
            import statsmodels.formula.api as smf

            # Create formula for regression (handle categorical variables properly)
            formula_parts = []
            for pred in predictors:
                if pred == "sex":
                    formula_parts.append(f"C({pred})")
                else:
                    formula_parts.append(pred)

            formula = "cluster_consistency ~ " + " + ".join(formula_parts)

            # Fit the model
            reg_model = smf.ols(formula, data=consistency_df).fit()

            # Save full regression results
            with open(stats_dir / "consistency_regression.txt", "w") as f:
                f.write(reg_model.summary().as_text())

            # Extract key findings
            results["regression_r2"] = reg_model.rsquared
            results["regression_r2_adj"] = reg_model.rsquared_adj

            # Find significant predictors (p < 0.05)
            sig_predictors = []
            for var, p_value in reg_model.pvalues.items():
                if float(p_value) < 0.05 and var != "Intercept":
                    coef = reg_model.params[var]
                    sig_predictors.append((var, coef, p_value))
                    significant_findings.append(
                        f"Consistency predictor: {var} (coef={coef:.3f}, p={p_value:.3f})"
                    )

            results["significant_predictors"] = sig_predictors

            # Create coefficient plot
            plt.figure(figsize=(10, 6))
            coefs = reg_model.params.drop("Intercept", errors="ignore")
            errors = reg_model.bse.drop("Intercept", errors="ignore")

            # Sort by coefficient magnitude
            coef_order = coefs.abs().sort_values(ascending=False).index
            coefs = coefs.reindex(coef_order)
            errors = errors.reindex(coef_order)

            colors = [
                "green" if p < 0.05 else "gray"
                for p in reg_model.pvalues.drop("Intercept", errors="ignore").reindex(
                    coef_order
                )
            ]

            plt.barh(range(len(coefs)), coefs, xerr=errors, color=colors, alpha=0.7)
            plt.yticks(range(len(coefs)), coefs.index)
            plt.axvline(x=0, color="black", linestyle="-", alpha=0.3)
            plt.title("Regression Coefficients for Cluster Consistency Predictors")
            plt.xlabel("Coefficient Value")
            plt.tight_layout()
            plt.savefig(stats_dir / "consistency_coefficients.png", dpi=300)
            plt.close()

        except Exception as e:
            print(f"Error in regression analysis: {e}")

    # 2. Correlation analysis with significance testing and correction
    print("Performing correlation analysis with multiple testing correction...")
    try:
        from scipy.stats import spearmanr
        from statsmodels.stats.multitest import multipletests

        # Select continuous variables for correlation analysis
        cont_vars = [
            p for p in predictors if p != "sex" and p in consistency_df.columns
        ]

        if cont_vars:
            # Calculate correlation matrix
            corr_matrix = consistency_df[cont_vars + ["cluster_consistency"]].corr(
                method="spearman"
            )

            # Extract correlations with consistency
            consistency_corrs = corr_matrix.loc[cont_vars, "cluster_consistency"]

            # Calculate p-values for each correlation
            p_values = []
            for var in cont_vars:
                corr, p = spearmanr(
                    consistency_df[var], consistency_df["cluster_consistency"]
                )
                p_values.append(p)

            # Apply multiple testing correction
            reject, p_adjusted, _, _ = multipletests(p_values, method="fdr_bh")

            # Combine results
            corr_results = pd.DataFrame(
                {
                    "variable": cont_vars,
                    "correlation": consistency_corrs.values,
                    "p_value": p_values,
                    "p_adjusted": p_adjusted,
                    "significant": reject,
                }
            )

            # Sort by absolute correlation
            corr_results = corr_results.sort_values(
                "correlation", key=abs, ascending=False
            )

            # Save to CSV
            corr_results.to_csv(stats_dir / "consistency_correlations.csv", index=False)

            # Add significant correlations to findings
            for _, row in corr_results[corr_results["significant"]].iterrows():
                significant_findings.append(
                    f"Correlation: {row['variable']} with consistency (r={row['correlation']:.3f}, adj_p={row['p_adjusted']:.3f})"
                )

            # Create correlation heatmap
            plt.figure(figsize=(10, 8))
            mask = np.zeros_like(corr_matrix, dtype=bool)
            mask[np.triu_indices_from(mask)] = True

            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                mask=mask,
                vmin=-1,
                vmax=1,
                center=0,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
            )

            plt.title("Correlation Matrix (Spearman)")
            plt.tight_layout()
            plt.savefig(stats_dir / "consistency_correlation_matrix.png", dpi=300)
            plt.close()

            # Store results
            results["correlations"] = corr_results.to_dict(orient="records")

    except Exception as e:
        print(f"Error in correlation analysis: {e}")

    # 3. T-tests/ANOVA for categorical variables
    print("Testing differences in consistency by categorical variables...")

    # For now, just examine sex differences if available
    if "sex" in consistency_df.columns:
        try:
            from scipy.stats import ttest_ind

            # Filter out any NaN values
            df_sex = consistency_df.dropna(subset=["sex", "cluster_consistency"])

            if df_sex["sex"].nunique() > 1:
                # Split by sex
                groups = df_sex.groupby("sex")["cluster_consistency"]

                # Perform t-test
                # Using only the first two categories if there are more than two
                group_labels = list(groups.groups.keys())[:2]
                if len(group_labels) == 2:
                    try:
                        t_result = ttest_ind(
                            groups.get_group(group_labels[0]),
                            groups.get_group(group_labels[1]),
                            equal_var=False,  # Using Welch's t-test for unequal variances
                        )
                        t_stat = float(t_result.statistic)
                        p_value = float(t_result.pvalue)
                    except AttributeError:
                        # Fallback for older scipy versions that might return a tuple
                        t_stat_val, p_value_val = ttest_ind(
                            groups.get_group(group_labels[0]),
                            groups.get_group(group_labels[1]),
                            equal_var=False,
                        )
                        t_stat = float(t_stat_val)
                        p_value = float(p_value_val)

                    # Save results
                    with open(stats_dir / "consistency_by_sex.txt", "w") as f:
                        f.write(f"T-test for cluster consistency by sex\n")
                        f.write(
                            f"Group 1 ({group_labels[0]}): n={len(groups.get_group(group_labels[0]))}, "
                        )
                        f.write(
                            f"mean={groups.get_group(group_labels[0]).mean():.3f}\n"
                        )
                        f.write(
                            f"Group 2 ({group_labels[1]}): n={len(groups.get_group(group_labels[1]))}, "
                        )
                        f.write(
                            f"mean={groups.get_group(group_labels[1]).mean():.3f}\n"
                        )
                        f.write(f"t-statistic: {t_stat:.3f}, p-value: {p_value:.3f}\n")

                    # Create visualization
                    plt.figure(figsize=(8, 6))
                    sns.boxplot(x="sex", y="cluster_consistency", data=df_sex)
                    plt.title(f"Cluster Consistency by Sex (p={p_value:.3f})")
                    plt.tight_layout()
                    plt.savefig(stats_dir / "consistency_by_sex.png", dpi=300)
                    plt.close()

                    # Add to findings if significant
                    if p_value < 0.05:
                        significant_findings.append(
                            f"Sex difference in consistency: {group_labels[0]}={groups.get_group(group_labels[0]).mean():.3f}, "
                            f"{group_labels[1]}={groups.get_group(group_labels[1]).mean():.3f} (p={p_value:.3f})"
                        )

        except Exception as e:
            print(f"Error in sex difference analysis: {e}")

    # 4. Compare consistency across modal clusters
    print("Analyzing consistency differences by modal cluster...")
    try:
        if "modal_cluster" in consistency_df.columns:
            from scipy.stats import kruskal

            # Group by modal cluster
            cluster_groups = []
            labels = []

            for cluster, group_df in consistency_df.groupby("modal_cluster"):
                if len(group_df) > 2:  # Need at least a few patients per cluster
                    cluster_groups.append(group_df["cluster_consistency"])
                    labels.append(cluster)

            if len(cluster_groups) > 1:
                # Perform Kruskal-Wallis test (non-parametric ANOVA)
                h_stat, p_value = kruskal(*cluster_groups)

                # Save results
                with open(stats_dir / "consistency_by_cluster.txt", "w") as f:
                    f.write(
                        f"Kruskal-Wallis test for cluster consistency by modal cluster\n"
                    )
                    f.write(f"H-statistic: {h_stat:.3f}, p-value: {p_value:.3f}\n\n")
                    f.write("Cluster consistency by modal cluster:\n")

                    for i, (label, group) in enumerate(zip(labels, cluster_groups)):
                        f.write(
                            f"Cluster {label}: n={len(group)}, mean={group.mean():.3f}, median={group.median():.3f}\n"
                        )

                # Create visualization
                plt.figure(figsize=(10, 6))

                # Use actual data for boxplot
                temp_df = pd.DataFrame(
                    {
                        "modal_cluster": [
                            str(l)
                            for l, g in zip(labels, cluster_groups)
                            for _ in range(len(g))
                        ],
                        "consistency": pd.concat(cluster_groups).values,
                    }
                )

                sns.boxplot(x="modal_cluster", y="consistency", data=temp_df)
                plt.title(f"Cluster Consistency by Modal Cluster (p={p_value:.3f})")
                plt.xlabel("Modal Cluster")
                plt.ylabel("Consistency Score")

                # Add mean labels
                for i, (_, group) in enumerate(zip(labels, cluster_groups)):
                    plt.text(
                        i, group.mean(), f"{group.mean():.2f}", ha="center", va="bottom"
                    )

                plt.tight_layout()
                plt.savefig(stats_dir / "consistency_by_modal_cluster.png", dpi=300)
                plt.close()

                # Add to findings if significant
                if p_value < 0.05:
                    significant_findings.append(
                        f"Modal cluster difference in consistency (p={p_value:.3f})"
                    )

                    # Perform post-hoc pairwise tests if significant
                    if p_value < 0.05 and len(cluster_groups) > 2:
                        try:
                            from scikit_posthocs import posthoc_dunn

                            # Create DataFrame for post-hoc test
                            posthoc_df = pd.DataFrame(
                                {
                                    "cluster": [
                                        str(l)
                                        for l, g in zip(labels, cluster_groups)
                                        for _ in range(len(g))
                                    ],
                                    "consistency": pd.concat(cluster_groups).values,
                                }
                            )

                            # Perform Dunn's test
                            posthoc = posthoc_dunn(
                                posthoc_df, val_col="consistency", group_col="cluster"
                            )

                            # Save to file
                            with open(stats_dir / "consistency_posthoc.txt", "w") as f:
                                f.write("Post-hoc Dunn's test results (p-values):\n")
                                f.write(posthoc.round(4).to_string())

                            # Extract significant pairwise differences
                            for i in range(len(labels)):
                                for j in range(i + 1, len(labels)):
                                    p_val_from_df = posthoc.iloc[i, j]
                                    if (
                                        pd.api.types.is_number(p_val_from_df)
                                        and float(p_val_from_df) < 0.05
                                    ):
                                        significant_findings.append(
                                            f"Clusters {labels[i]} vs {labels[j]} differ in consistency (p={float(p_val_from_df):.3f})"
                                        )
                        except Exception as e:
                            print(f"Error in post-hoc analysis: {e}")

    except Exception as e:
        print(f"Error in modal cluster analysis: {e}")

    # Save all significant findings to file
    with open(stats_dir / "significant_findings.txt", "w") as f:
        f.write("Significant Findings for Cluster Consistency:\n")
        f.write("=" * 50 + "\n\n")

        if significant_findings:
            for i, finding in enumerate(significant_findings, 1):
                f.write(f"{i}. {finding}\n")
        else:
            f.write("No significant findings detected.\n")

    # Return results
    results["significant_findings"] = significant_findings
    return results


def analyze_clusters_inferential(eps_df, out_dir):
    """
    Perform detailed inferential statistical analysis on clusters.

    Parameters
    ----------
    eps_df : pd.DataFrame
        DataFrame with episodes and cluster assignments
    out_dir : Path
        Output directory for saving results

    Returns
    -------
    dict
        Dictionary with key statistical findings
    """
    # Use only the main clusters
    cluster_col = "cluster"
    cluster_type = "Cluster"

    # Create output directory
    stats_dir = out_dir / "stats" / "inferential"
    stats_dir.mkdir(exist_ok=True, parents=True)

    results = {}
    significant_findings = []

    # Filter out noise points
    df_valid = eps_df[eps_df[cluster_col] != -1].copy()

    # Check if we have at least 2 clusters for comparison
    n_clusters = df_valid[cluster_col].nunique()
    if n_clusters < 2:
        print(
            f"Not enough clusters for inferential analysis (found {n_clusters}, need at least 2)"
        )
        return {}

    print(
        f"\nPerforming inferential analysis for {cluster_type.lower()} characteristics..."
    )

    # Identify variables that were used in clustering (should be excluded from inferential)
    # Check the parameters.txt file to identify the clustering algorithm
    params_file = out_dir / "parameters.txt"
    excluded_vars = []

    if params_file.exists():
        try:
            with open(params_file, "r") as f:
                params_text = f.read()

            # Extract clustering algorithm
            import re

            algorithm_match = re.search(r"clustering_algorithm: (.+)", params_text)
            nadir_weight_match = re.search(r"nadir_weight: (.+)", params_text)

            if algorithm_match:
                algorithm = algorithm_match.group(1)

                # If nadir weight > 0, exclude nadir timing variables
                if nadir_weight_match and float(nadir_weight_match.group(1)) > 0:
                    excluded_vars.extend(
                        [
                            "nadir_time_pct",
                            "nadir_time_hours",
                            "time_to_nadir_normalized",
                        ]
                    )
                    print(
                        f"Excluding nadir timing variables from inferential statistics as they were used in clustering"
                    )
        except Exception as e:
            print(f"Error reading parameters file: {e}")

    if not excluded_vars:
        # Default exclusions if we can't determine from parameters file
        excluded_vars = ["nadir_time_pct"]
        print(f"Using default exclusions for inferential statistics: {excluded_vars}")

    # 1. Cluster validation metrics
    print("Calculating cluster validation metrics...")
    try:
        from sklearn.metrics import (
            silhouette_score,
            calinski_harabasz_score,
            davies_bouldin_score,
        )

        # We need numeric features for validation metrics
        # Use UMAP coordinates if available
        if "umap1" in df_valid.columns and "umap2" in df_valid.columns:
            X_validation = df_valid[["umap1", "umap2"]].values

            # Calculate validation metrics
            try:
                s_score = silhouette_score(X_validation, df_valid[cluster_col])
                ch_score = calinski_harabasz_score(X_validation, df_valid[cluster_col])
                db_score = davies_bouldin_score(X_validation, df_valid[cluster_col])

                # Save results
                with open(stats_dir / "cluster_validation.txt", "w") as f:
                    f.write(f"{cluster_type} Validation Metrics:\n")
                    f.write(f"Number of {cluster_type.lower()}s: {n_clusters}\n")
                    f.write(
                        f"Silhouette Score: {s_score:.3f} (higher is better, range [-1, 1])\n"
                    )
                    f.write(
                        f"Calinski-Harabasz Score: {ch_score:.3f} (higher is better)\n"
                    )
                    f.write(f"Davies-Bouldin Score: {db_score:.3f} (lower is better)\n")

                # Store results
                results["validation"] = {
                    "silhouette": s_score,
                    "calinski_harabasz": ch_score,
                    "davies_bouldin": db_score,
                }
            except Exception as e:
                print(f"Error calculating validation metrics: {e}")
    except ImportError:
        print("sklearn metrics not available for cluster validation")

    # 2. Discriminant analysis - which variables best separate the clusters
    print("Performing discriminant analysis to identify key cluster variables...")
    try:
        # Variables to analyze
        cont_vars = [
            "age",
            "patient_chronotype",
            "abs_chronotype_desync",
            "chronotype_desync",
            "se",
            "waso",
            "duration_secs",
            "episode_hr_mean",
            "nadir_time_pct",
            "nadir_hr",
            "sfi",
        ]

        # Filter out excluded variables
        cont_vars = [var for var in cont_vars if var not in excluded_vars]

        # Filter to available columns
        available_vars = [var for var in cont_vars if var in df_valid.columns]

        if available_vars and len(available_vars) >= 2:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler, LabelEncoder

            # Prepare data - handle NaNs and standardize
            X = df_valid[available_vars].copy()
            X = X.fillna(X.mean())

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=available_vars)

            # Encode target
            le = LabelEncoder()
            y = le.fit_transform(df_valid[cluster_col])

            # Train Random Forest for feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_scaled, y)

            # Get feature importances
            importances = rf.feature_importances_

            # Create DataFrame of importances
            importance_df = pd.DataFrame(
                {"feature": available_vars, "importance": importances}
            ).sort_values("importance", ascending=False)

            # Save to CSV
            importance_df.to_csv(stats_dir / "feature_importance.csv", index=False)

            # Create importance plot
            plt.figure(figsize=(10, 6))
            sns.barplot(x="importance", y="feature", data=importance_df)
            plt.title(f"Feature Importance for {cluster_type} Discrimination")
            plt.tight_layout()
            plt.savefig(stats_dir / "feature_importance.png", dpi=300)
            plt.close()

            # Add top features to findings
            top_features = importance_df.head(3)["feature"].tolist()
            importance_values = importance_df.head(3)["importance"].tolist()

            significant_findings.append(
                f"Top features distinguishing {cluster_type.lower()}s: "
                + ", ".join(
                    [f"{f} ({v:.3f})" for f, v in zip(top_features, importance_values)]
                )
            )

            # Store results
            results["feature_importance"] = importance_df.to_dict(orient="records")

    except Exception as e:
        print(f"Error in discriminant analysis: {e}")

    # 3. Detailed ANOVA testing for differences between clusters
    print("Performing detailed ANOVA tests between clusters...")
    try:
        from scipy.stats import f_oneway

        # Variables to test
        test_vars = [
            "age",
            "patient_chronotype",
            "abs_chronotype_desync",
            "chronotype_desync",
            "se",
            "waso",
            "duration_secs",
            "episode_hr_mean",
            "nadir_time_pct",
            "nadir_hr",
            "chronotype_cont",
        ]

        # Filter out excluded variables
        test_vars = [var for var in test_vars if var not in excluded_vars]

        # Filter to available columns
        available_vars = [var for var in test_vars if var in df_valid.columns]

        if available_vars:
            # Store ANOVA results
            anova_results = []

            for var in available_vars:
                try:
                    # Group data by cluster
                    groups = []
                    for cluster in df_valid[cluster_col].unique():
                        group_data = df_valid[df_valid[cluster_col] == cluster][
                            var
                        ].dropna()
                        if len(group_data) > 0:
                            groups.append(group_data)

                    if len(groups) > 1:
                        # Run ANOVA
                        f_stat, p_value = f_oneway(*groups)

                        # Store result
                        result = {
                            "variable": var,
                            "f_statistic": f_stat,
                            "p_value": p_value,
                            "significant": p_value < 0.05,
                        }
                        anova_results.append(result)

                        # Add to findings if significant
                        if p_value < 0.05:
                            significant_findings.append(
                                f"{var} differs significantly between {cluster_type.lower()}s (F={f_stat:.3f}, p={p_value:.3f})"
                            )
                except Exception as e:
                    print(f"Error in ANOVA for {var}: {e}")

            # Create summary DataFrame
            if anova_results:
                anova_df = pd.DataFrame(anova_results)

                # Apply multiple testing correction
                from statsmodels.stats.multitest import multipletests

                _, p_adjusted, _, _ = multipletests(
                    anova_df["p_value"], method="fdr_bh"
                )
                anova_df["p_adjusted"] = p_adjusted
                anova_df["significant_adjusted"] = p_adjusted < 0.05

                # Sort by significance
                anova_df = anova_df.sort_values("p_value")

                # Save to CSV
                anova_df.to_csv(stats_dir / "anova_results.csv", index=False)

                # Create visualization of significant differences
                sig_vars = anova_df[anova_df["significant_adjusted"]][
                    "variable"
                ].tolist()

                if sig_vars:
                    # Create plots for each significant variable
                    for i, var in enumerate(sig_vars):
                        plt.figure(figsize=(10, 6))
                        sns.boxplot(x=cluster_col, y=var, data=df_valid)
                        p_val = anova_df[anova_df["variable"] == var][
                            "p_adjusted"
                        ].iloc[0]
                        formatted_p_val = _format_p_value_for_title(float(p_val))
                        plt.title(f"{var} by {cluster_type} (adj. {formatted_p_val})")
                        plt.xlabel(cluster_type)
                        plt.ylabel(var)

                        # Add mean values
                        means = df_valid.groupby(cluster_col)[var].mean()
                        for j, cat in enumerate(means.index):
                            mean_val = means[cat].item()
                            plt.text(
                                j, mean_val, f"{mean_val:.2f}", ha="center", va="bottom"
                            )

                        plt.tight_layout()
                        plt.savefig(stats_dir / f"{var}_by_cluster.png", dpi=300)
                        plt.close()

                # Store results
                results["anova"] = anova_df.to_dict(orient="records")

    except Exception as e:
        print(f"Error in ANOVA testing: {e}")

    # 4. Chi-squared tests for categorical variables (e.g., sex)
    if "sex" in df_valid.columns:
        print("Testing categorical associations with clusters...")
        try:
            from scipy.stats import chi2_contingency

            # Create contingency table
            crosstab = pd.crosstab(df_valid[cluster_col], df_valid["sex"])

            # Run chi-squared test
            chi2, p, dof, expected = chi2_contingency(crosstab)

            # Save results
            with open(stats_dir / "categorical_associations.txt", "w") as f:
                f.write(
                    f"Chi-squared test for sex distribution across {cluster_type.lower()}s:\n"
                )
                f.write(f"Chi2 = {chi2:.3f}, p = {p:.3f}, df = {dof}\n\n")
                f.write("Contingency table:\n")
                f.write(crosstab.to_string())
                f.write("\n\nExpected frequencies:\n")
                f.write(
                    pd.DataFrame(
                        expected,
                        index=crosstab.index,
                        columns=crosstab.columns,
                    ).to_string()
                )

            # Create visualization
            plt.figure(figsize=(10, 6))
            crosstab_norm = crosstab.div(crosstab.sum(axis=1), axis=0)
            crosstab_norm.plot(kind="bar", stacked=True)
            plt.title(f"Sex Distribution by {cluster_type} (p={p:.3f})")
            plt.xlabel(cluster_type)
            plt.ylabel("Proportion")
            plt.tight_layout()
            plt.savefig(stats_dir / "sex_by_cluster.png", dpi=300)
            plt.close()

            # Add to findings if significant
            current_p_value = float(p)
            if current_p_value < 0.05:
                significant_findings.append(
                    f"Sex distribution differs significantly across {cluster_type.lower()}s (p={current_p_value:.3f})"
                )

            # Store results
            results["categorical"] = {"chi2": chi2, "p_value": p, "dof": dof}

        except Exception as e:
            print(f"Error in categorical association testing: {e}")

    # Save all significant findings to file
    with open(stats_dir / "significant_findings.txt", "w") as f:
        f.write(f"Significant Findings for {cluster_type}s:\n")
        f.write("=" * 50 + "\n\n")

        if significant_findings:
            for i, finding in enumerate(significant_findings, 1):
                f.write(f"{i}. {finding}\n")
        else:
            f.write("No significant findings detected.\n")

    # Return results
    results["significant_findings"] = significant_findings
    return results


def analyze_hr_chronotype_relationships(eps_df, out_dir):
    """
    Analyze relationships between HR metrics, chronotype desynchronization, and sleep indices.

    Parameters
    ----------
    eps_df : pd.DataFrame
        DataFrame with episode data
    out_dir : Path
        Output directory

    Returns
    -------
    dict
        Dictionary of results and findings
    """
    # Create output directory
    stats_dir = out_dir / "stats" / "hr_chrono_analysis"
    stats_dir.mkdir(exist_ok=True, parents=True)

    results = {}
    significant_findings = []

    # Filter out noise points
    cluster_col = "cluster"
    df_valid = eps_df[eps_df[cluster_col] != -1].copy()

    print(
        "\nAnalyzing heart rate metrics, chronotype relationships, and sleep indices..."
    )

    try:
        # Define metrics to analyze
        hr_metrics = ["nadir_hr", "episode_hr_mean"]
        if "resting_hr" in df_valid.columns:
            hr_metrics.append("resting_hr")

        chrono_metrics = [
            "chronotype_desync",
            "abs_chronotype_desync",
            "patient_chronotype",
        ]

        sleep_indices = ["se", "waso", "sfi", "duration_secs", "sri"]
        if "activity_idx" in df_valid.columns:
            sleep_indices.append("activity_idx")

        # Filter available metrics
        available_hr = [col for col in hr_metrics if col in df_valid.columns]
        available_chrono = [col for col in chrono_metrics if col in df_valid.columns]
        available_sleep = [col for col in sleep_indices if col in df_valid.columns]

        # 1. Correlation matrix between all HR, chronotype, and nadir timing variables
        analysis_vars = available_hr + available_chrono + ["nadir_time_pct"]
        analysis_vars = list(set([v for v in analysis_vars if v in df_valid.columns]))

        if len(analysis_vars) >= 2:
            # Calculate correlation matrix
            corr_matrix = df_valid[analysis_vars].corr(method="pearson")

            # Save correlation matrix
            corr_matrix.to_csv(stats_dir / "hr_chrono_correlation_matrix.csv")

            # Create heatmap visualization
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(
                corr_matrix,
                mask=mask,
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                center=0,
                annot=True,
                fmt=".2f",
                linewidths=0.5,
            )
            plt.title("Correlation Matrix: HR Metrics, Chronotype, and Nadir Timing")
            plt.tight_layout()
            plt.savefig(stats_dir / "hr_chrono_correlation_heatmap.png", dpi=300)
            plt.close()

            # Extract significant correlations
            sig_correlations = []
            for i in range(len(analysis_vars)):
                for j in range(i + 1, len(analysis_vars)):
                    var1, var2 = analysis_vars[i], analysis_vars[j]
                    # Calculate p-value for this correlation
                    corr_coef = corr_matrix.loc[var1, var2]
                    # Skip if correlation is NaN
                    if pd.isna(corr_coef):
                        continue

                    # Calculate p-value
                    n = len(df_valid[[var1, var2]].dropna())
                    if n > 5:  # Only if we have enough data points
                        t_stat = corr_coef * np.sqrt((n - 2) / (1 - corr_coef**2))
                        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))

                        if p_value < 0.05:
                            sig_correlations.append(
                                {
                                    "variable1": var1,
                                    "variable2": var2,
                                    "correlation": corr_coef,
                                    "p_value": p_value,
                                    "n": n,
                                }
                            )
                            significant_findings.append(
                                f"Significant correlation between {var1} and {var2}: r={corr_coef:.3f}, p={p_value:.4f}, n={n}"
                            )

            # Save significant correlations
            if sig_correlations:
                pd.DataFrame(sig_correlations).to_csv(
                    stats_dir / "significant_hr_chrono_correlations.csv", index=False
                )

                # Create scatter plots for top significant correlations
                top_corrs = sorted(
                    sig_correlations, key=lambda x: abs(x["correlation"]), reverse=True
                )[:5]
                for corr_info in top_corrs:
                    var1, var2 = corr_info["variable1"], corr_info["variable2"]
                    plt.figure(figsize=(8, 6))
                    sns.scatterplot(
                        x=var1, y=var2, data=df_valid, hue=cluster_col, palette="deep"
                    )
                    plt.title(
                        f"Correlation: {var1} vs {var2}\nr={corr_info['correlation']:.3f}, p={corr_info['p_value']:.4f}"
                    )

                    # Add regression line
                    sns.regplot(
                        x=var1,
                        y=var2,
                        data=df_valid,
                        scatter=False,
                        line_kws={"color": "red", "linestyle": "--"},
                    )

                    plt.tight_layout()
                    plt.savefig(stats_dir / f"{var1}_{var2}_correlation.png", dpi=300)
                    plt.close()

        # 2. Analysis of sleep indices and nadir timing
        if "nadir_time_pct" in df_valid.columns and available_sleep:
            sleep_nadir_results = []

            for sleep_var in available_sleep:
                try:
                    # Get clean data
                    clean_data = df_valid[[sleep_var, "nadir_time_pct"]].dropna()

                    if len(clean_data) > 5:
                        # Calculate correlation
                        corr, p_value = stats.pearsonr(
                            clean_data["nadir_time_pct"], clean_data[sleep_var]
                        )

                        sleep_nadir_results.append(
                            {
                                "sleep_index": sleep_var,
                                "correlation": corr,
                                "p_value": p_value,
                                "n": len(clean_data),
                                "significant": float(p_value) < 0.05,
                            }
                        )

                        # Ensure p_value is float before comparison
                        if float(p_value) < 0.05:
                            significant_findings.append(
                                f"Sleep index {sleep_var} is significantly correlated with nadir timing: r={corr:.3f}, p={float(p_value):.4f}"
                            )

                            # Create scatter plot
                            plt.figure(figsize=(8, 6))
                            scatter = sns.scatterplot(
                                x="nadir_time_pct",
                                y=sleep_var,
                                data=df_valid,
                                hue=cluster_col,
                                palette="deep",
                            )

                            plt.title(
                                f"Nadir Timing vs {sleep_var}\nr={corr:.3f}, p={float(p_value):.4f}"
                            )
                            plt.xlabel("Nadir Timing (% of sleep period)")

                            # Add regression line
                            sns.regplot(
                                x="nadir_time_pct",
                                y=sleep_var,
                                data=df_valid,
                                scatter=False,
                                line_kws={"color": "red", "linestyle": "--"},
                            )

                            plt.tight_layout()
                            plt.savefig(
                                stats_dir / f"nadir_timing_{sleep_var}.png", dpi=300
                            )
                            plt.close()
                except Exception as e:
                    print(f"Error analyzing nadir timing and {sleep_var}: {e}")

            # Save results for sleep indices and nadir timing
            if sleep_nadir_results:
                pd.DataFrame(sleep_nadir_results).to_csv(
                    stats_dir / "sleep_indices_nadir_timing.csv", index=False
                )

                # Create summary bar plot
                sleep_results_df = pd.DataFrame(sleep_nadir_results)
                if not sleep_results_df.empty:
                    plt.figure(figsize=(10, 6))
                    bars = sns.barplot(
                        x="sleep_index",
                        y="correlation",
                        data=sleep_results_df,
                        hue="significant",
                        palette=["gray", "blue"],
                    )
                    plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)
                    plt.title("Correlation between Sleep Indices and Nadir Timing")
                    plt.xlabel("Sleep Index")
                    plt.ylabel("Correlation Coefficient (r)")
                    plt.xticks(rotation=45)

                    # Add correlation values
                    for i, p in enumerate(sleep_results_df["p_value"]):
                        stars = "**" if p < 0.01 else "*" if p < 0.05 else ""
                        bars.text(
                            i,
                            sleep_results_df["correlation"].iloc[i],
                            f"{sleep_results_df['correlation'].iloc[i]:.2f}{stars}",
                            ha="center",
                        )

                    plt.tight_layout()
                    plt.savefig(stats_dir / "sleep_indices_nadir_summary.png", dpi=300)
                    plt.close()

        # 3. Compare HR metrics across clusters
        if available_hr:
            # Store ANOVA results
            hr_anova_results = []

            for hr_var in available_hr:
                try:
                    # Group data by cluster
                    groups = []
                    cluster_ids = []
                    for cluster in df_valid[cluster_col].unique():
                        group_data = df_valid[df_valid[cluster_col] == cluster][
                            hr_var
                        ].dropna()
                        if len(group_data) > 0:
                            groups.append(group_data)
                            cluster_ids.append(cluster)

                    if len(groups) > 1:
                        # Run ANOVA
                        f_stat, p_value = stats.f_oneway(*groups)

                        # Store result
                        result = {
                            "hr_metric": hr_var,
                            "f_statistic": f_stat,
                            "p_value": p_value,
                            "significant": p_value < 0.05,
                        }
                        hr_anova_results.append(result)

                        if p_value < 0.05:
                            significant_findings.append(
                                f"HR metric {hr_var} differs significantly between clusters: F={f_stat:.3f}, p={p_value:.4f}"
                            )

                            # Create boxplot
                            plt.figure(figsize=(10, 6))
                            sns.boxplot(x=cluster_col, y=hr_var, data=df_valid)
                            plt.title(f"{hr_var} by Cluster (p={p_value:.4f})")

                            # Add cluster means
                            means = [g.mean() for g in groups]
                            for i, mean_val in enumerate(means):
                                plt.text(
                                    i,
                                    mean_val,
                                    f"{mean_val:.1f}",
                                    ha="center",
                                    va="bottom",
                                )

                            plt.tight_layout()
                            plt.savefig(stats_dir / f"{hr_var}_by_cluster.png", dpi=300)
                            plt.close()

                            # If significant, run post-hoc tests
                            if p_value < 0.05 and len(groups) > 2:
                                try:
                                    from scikit_posthocs import posthoc_tukey

                                    # Create DataFrame for post-hoc
                                    posthoc_data = []
                                    for i, (cluster, group) in enumerate(
                                        zip(cluster_ids, groups)
                                    ):
                                        for val in group:
                                            posthoc_data.append(
                                                {"cluster": cluster, "value": val}
                                            )

                                    posthoc_df = pd.DataFrame(posthoc_data)

                                    # Perform Tukey's test
                                    posthoc_results = posthoc_tukey(
                                        posthoc_df, val_col="value", group_col="cluster"
                                    )

                                    # Save results
                                    posthoc_results.to_csv(
                                        stats_dir / f"{hr_var}_posthoc.csv"
                                    )
                                except Exception as e:
                                    print(
                                        f"Error in post-hoc testing for {hr_var}: {e}"
                                    )
                except Exception as e:
                    print(f"Error in ANOVA for {hr_var}: {e}")

            # Save HR ANOVA results
            if hr_anova_results:
                pd.DataFrame(hr_anova_results).to_csv(
                    stats_dir / "hr_metrics_anova.csv", index=False
                )

    except Exception as e:
        print(f"Error in HR-chronotype relationship analysis: {e}")

    # Save significant findings
    with open(stats_dir / "significant_findings.txt", "w") as f:
        f.write(
            "Significant Findings for HR, Chronotype, and Sleep Indices Analysis:\n"
        )
        f.write("=" * 70 + "\n\n")

        if significant_findings:
            for i, finding in enumerate(significant_findings, 1):
                f.write(f"{i}. {finding}\n")
        else:
            f.write("No significant findings detected.\n")

    results["findings"] = significant_findings
    return results


def main():
    """Main entry point for the sleep analysis pipeline."""
    # Get default parameters
    params = get_default_params()

    # Define the base output directory
    base_out_dir = Path(params["out_dir"])
    base_out_dir.mkdir(parents=True, exist_ok=True)

    # Load the raw data once
    print(f"Loading data from {params['data_path']}...")
    df_raw = pd.read_pickle(params["data_path"])

    # Create a summary file to track all results
    summary_path = base_out_dir / "grid_search_summary.csv"
    with open(summary_path, "w") as f:
        f.write(
            "equalize_method,eps,min_samples,nadir_weight,clustering_algorithm,n_clusters,noise_points,valid_points,min_clusters,max_clusters\n"
        )

    # Generate parameter combinations based on algorithm type
    param_combinations = []

    for eq_method in params["equalize_method_values"]:
        for algorithm in params["clustering_algorithm_values"]:
            # For each algorithm, only vary relevant parameters
            if algorithm == "tda_nadir_umap_kmeans_silhouette_k":
                # Only vary nadir_weight and k ranges
                for nadir_weight in params["nadir_weight_values"]:
                    param_combinations.append(
                        (
                            eq_method,
                            params["eps_values"][0],  # Use default eps (not used)
                            params["min_samples_values"][
                                0
                            ],  # Use default min_samples (not used)
                            nadir_weight,
                            algorithm,
                            params["min_clusters"],
                            params["max_clusters"],
                        )
                    )
            elif algorithm == "gmm_paa_hr_bic":
                # GMM: vary nadir_weight and k ranges
                for nadir_weight in params["nadir_weight_values"]:
                    param_combinations.append(
                        (
                            eq_method,
                            params["eps_values"][0],  # Not used
                            params["min_samples_values"][0],  # Not used
                            nadir_weight,
                            algorithm,
                            params["min_clusters"],
                            params["max_clusters"],
                        )
                    )
            elif algorithm == "hac_dtw":
                # HAC with DTW: vary nadir_weight and k ranges
                for nadir_weight in params["nadir_weight_values"]:
                    param_combinations.append(
                        (
                            eq_method,
                            params["eps_values"][0],  # Not used
                            params["min_samples_values"][0],  # Not used
                            nadir_weight,
                            algorithm,
                            params["min_clusters"],
                            params["max_clusters"],
                        )
                    )
            elif algorithm == "spectral_clustering_paa_hr":
                # Spectral clustering: vary nadir_weight and k ranges
                for nadir_weight in params["nadir_weight_values"]:
                    param_combinations.append(
                        (
                            eq_method,
                            params["eps_values"][0],  # Not used
                            params["min_samples_values"][0],  # Not used
                            nadir_weight,
                            algorithm,
                            params["min_clusters"],
                            params["max_clusters"],
                        )
                    )
            elif algorithm == "gmm_slope_bic":
                # GMM on slope features: vary nadir_weight and k ranges
                for nadir_weight in params["nadir_weight_values"]:
                    param_combinations.append(
                        (
                            eq_method,
                            params["eps_values"][0],  # Not used
                            params["min_samples_values"][0],  # Not used
                            nadir_weight,
                            algorithm,
                            params["min_clusters"],
                            params["max_clusters"],
                        )
                    )
            elif algorithm == "hac_dtw_slope":
                # HAC with DTW on slope features: vary nadir_weight and k ranges
                for nadir_weight in params["nadir_weight_values"]:
                    param_combinations.append(
                        (
                            eq_method,
                            params["eps_values"][0],  # Not used
                            params["min_samples_values"][0],  # Not used
                            nadir_weight,
                            algorithm,
                            params["min_clusters"],
                            params["max_clusters"],
                        )
                    )
            elif algorithm == "spectral_clustering_slope":
                # Spectral clustering on slope features: vary nadir_weight and k ranges
                for nadir_weight in params["nadir_weight_values"]:
                    param_combinations.append(
                        (
                            eq_method,
                            params["eps_values"][0],  # Not used
                            params["min_samples_values"][0],  # Not used
                            nadir_weight,
                            algorithm,
                            params["min_clusters"],
                            params["max_clusters"],
                        )
                    )
            elif algorithm == "tda_nadir_umap_kmeans_3":
                # K-means with fixed k=3 on UMAP embedding
                for nadir_weight in params["nadir_weight_values"]:
                    param_combinations.append(
                        (
                            eq_method,
                            params["eps_values"][0],  # Not used
                            params["min_samples_values"][0],  # Not used
                            nadir_weight,
                            algorithm,
                            params["min_clusters"],
                            params["max_clusters"],
                        )
                    )
            elif algorithm == "gmm_paa_hr_3":
                # GMM with fixed k=3 on PAA-HR features
                for nadir_weight in params["nadir_weight_values"]:
                    param_combinations.append(
                        (
                            eq_method,
                            params["eps_values"][0],  # Not used
                            params["min_samples_values"][0],  # Not used
                            nadir_weight,
                            algorithm,
                            params["min_clusters"],
                            params["max_clusters"],
                        )
                    )
            elif algorithm == "hac_dtw_3":
                # Hierarchical Agglomerative Clustering with DTW and fixed k=3
                for nadir_weight in params["nadir_weight_values"]:
                    param_combinations.append(
                        (
                            eq_method,
                            params["eps_values"][0],  # Not used
                            params["min_samples_values"][0],  # Not used
                            nadir_weight,
                            algorithm,
                            params["min_clusters"],
                            params["max_clusters"],
                        )
                    )
            elif algorithm == "spectral_clustering_paa_hr_3":
                # Spectral Clustering with fixed k=3 on PAA-HR features
                for nadir_weight in params["nadir_weight_values"]:
                    param_combinations.append(
                        (
                            eq_method,
                            params["eps_values"][0],  # Not used
                            params["min_samples_values"][0],  # Not used
                            nadir_weight,
                            algorithm,
                            params["min_clusters"],
                            params["max_clusters"],
                        )
                    )
            elif algorithm == "gmm_slope_3":
                # GMM on HR slope features with fixed k=3
                for nadir_weight in params["nadir_weight_values"]:
                    param_combinations.append(
                        (
                            eq_method,
                            params["eps_values"][0],  # Not used
                            params["min_samples_values"][0],  # Not used
                            nadir_weight,
                            algorithm,
                            params["min_clusters"],
                            params["max_clusters"],
                        )
                    )
            elif algorithm == "hac_dtw_slope_3":
                # Hierarchical Agglomerative Clustering with DTW on HR slope with fixed k=3
                for nadir_weight in params["nadir_weight_values"]:
                    param_combinations.append(
                        (
                            eq_method,
                            params["eps_values"][0],  # Not used
                            params["min_samples_values"][0],  # Not used
                            nadir_weight,
                            algorithm,
                            params["min_clusters"],
                            params["max_clusters"],
                        )
                    )
            elif algorithm == "spectral_clustering_slope_3":
                # Spectral Clustering on HR slope features with fixed k=3
                for nadir_weight in params["nadir_weight_values"]:
                    param_combinations.append(
                        (
                            eq_method,
                            params["eps_values"][0],  # Not used
                            params["min_samples_values"][0],  # Not used
                            nadir_weight,
                            algorithm,
                            params["min_clusters"],
                            params["max_clusters"],
                        )
                    )

    print(f"Total configurations to test: {len(param_combinations)}")

    # Run each parameter combination
    for i, (
        equalize_method,
        eps,
        min_samples,
        nadir_weight,
        clustering_algorithm,
        min_k,
        max_k,
    ) in enumerate(param_combinations):
        print(f"\nRunning configuration {i+1}/{len(param_combinations)}")

        # Create a subfolder for this configuration
        config_dir = (
            base_out_dir
            / f"eq_{equalize_method}_clust_{clustering_algorithm}_nw_{nadir_weight}_k_{min_k}-{max_k}"
        )

        # Add eps and min_samples to DBSCAN folder name
        if clustering_algorithm == "umap_dbscan":
            config_dir = (
                config_dir.parent
                / f"{config_dir.name}_eps_{eps}_minSamples_{min_samples}"
            )

        try:
            # Run the pipeline with this configuration
            stats = run_pipeline(
                df_raw=df_raw,
                out_dir=config_dir,
                equalize_method=equalize_method,
                eps=eps,
                min_samples=min_samples,
                nadir_weight=nadir_weight,
                use_causal=params["use_causal"],
                causal_exposure=params["causal_exposure"],
                prior_type=params["prior_type"],
                prior_scale=params["prior_scale"],
                compare_fudolig=params["compare_fudolig"],
                clustering_algorithm=clustering_algorithm,
                cv_method=params["cv_method"],
                min_clusters=min_k,
                max_clusters=max_k,
                favor_higher_k=params["favor_higher_k"],
                penalty_weight=params["penalty_weight"],
                reduced_mcmc=params["reduced_mcmc"],
            )

            # Only add to summary file if analysis wasn't stopped early
            if not stats.get("stopped_early", False):
                with open(summary_path, "a") as f:
                    f.write(
                        f"{equalize_method},{eps},{min_samples},{nadir_weight},{clustering_algorithm},{stats['n_clusters']},"
                        f"{stats['noise_points']},{stats['total_points']-stats['noise_points']},{min_k},{max_k}\n"
                    )

        except Exception as e:
            print(
                f"Error in configuration (eq={equalize_method}, algorithm={clustering_algorithm}, nadir_weight={nadir_weight}):"
            )
            print(f"  {str(e)}")

            # Log the error
            with open(base_out_dir / "grid_search_errors.txt", "a") as f:
                f.write(
                    f"Error in configuration (eq={equalize_method}, algorithm={clustering_algorithm}, nadir_weight={nadir_weight}):\n"
                )
                f.write(f"  {str(e)}\n\n")

    # Create summary visualizations
    if Path(summary_path).exists():
        summary_df = pd.read_csv(summary_path)
        create_grid_search_summary(
            base_out_dir,
            params["eps_values"],
            params["min_samples_values"],
            params["nadir_weight_values"],
            params["equalize_method_values"],
            params["clustering_algorithm_values"],
            summary_df,
        )

    print(f"\nGrid search complete! Results saved to {base_out_dir}")
    print(f"Summary file: {summary_path}")


if __name__ == "__main__":
    main()
