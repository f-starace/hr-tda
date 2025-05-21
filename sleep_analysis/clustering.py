"""
Clustering module for sleep data analysis.

This module contains functions for dimensionality reduction,
clustering, and meta-clustering of sleep heart rate patterns.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import umap
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy import stats
from sklearn.mixture import GaussianMixture
from tslearn.metrics import dtw
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
from sleep_analysis.features import slope_fixed_len


def find_optimal_clusters(
    X: np.ndarray,
    min_clusters: int = 2,
    max_clusters: int = 10,
    random_state: int = 42,
    favor_higher_k: bool = True,
    penalty_weight: float = 0.05,
) -> Tuple[int, float, List[float]]:
    """
    Find the optimal number of clusters using Silhouette Analysis with adjustments.

    Parameters
    ----------
    X : np.ndarray
        Input data for clustering
    min_clusters : int, default=2
        Minimum number of clusters to try
    max_clusters : int, default=10
        Maximum number of clusters to try
    random_state : int, default=42
        Random state for KMeans
    favor_higher_k : bool, default=True
        Whether to apply a small penalty to favor slightly higher cluster counts
    penalty_weight : float, default=0.05
        Strength of the penalty for lower k values (0 = no penalty)

    Returns
    -------
    Tuple[int, float, List[float]]
        Optimal number of clusters, best silhouette score, and list of all silhouette scores
    """
    # Ensure we have enough data points
    max_clusters = min(max_clusters, len(X) - 1)

    if max_clusters < min_clusters:
        print(f"Not enough data points for clustering. Using default k=2.")
        return 2, 0.0, [0.0]

    # Store silhouette scores
    silhouette_scores = []
    raw_scores = []  # Store original scores without penalty

    # Also calculate Calinski-Harabasz index for comparison
    ch_scores = []

    print(f"\nClustering quality analysis for k={min_clusters} to {max_clusters}:")
    print("-" * 60)
    print(f"{'k':^4}{'Silhouette':^15}{'Calinski-Harabasz':^20}{'Adjusted Score':^15}")
    print("-" * 60)

    # Try different numbers of clusters
    for k in range(min_clusters, max_clusters + 1):
        # Fit KMeans
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(X)

        # Calculate scores
        try:
            silhouette = silhouette_score(X, cluster_labels)
            ch_index = calinski_harabasz_score(X, cluster_labels)
            raw_scores.append(silhouette)
            ch_scores.append(ch_index)

            # Apply mild penalty to favor higher k (if enabled)
            if favor_higher_k and k > min_clusters:
                # Penalty decreases as k increases, scaled by penalty_weight
                # This slightly tilts the preference toward higher k values
                adjusted_score = silhouette * (
                    1
                    + penalty_weight
                    * (k - min_clusters)
                    / (max_clusters - min_clusters)
                )
            else:
                adjusted_score = silhouette

            silhouette_scores.append(adjusted_score)

            print(
                f"{k:^4}{silhouette:.4f}{'*' if silhouette == max(raw_scores) else ' ':^1}"
                f"{ch_index:^20.1f}{adjusted_score:.4f}{'*' if adjusted_score == max(silhouette_scores) else ' ':^1}"
            )

        except Exception as e:
            print(f"{k:^4}Error: {str(e)}")
            silhouette_scores.append(-1)
            raw_scores.append(-1)
            ch_scores.append(-1)

    # Find the best number of clusters (highest adjusted silhouette score)
    if not silhouette_scores or max(silhouette_scores) <= 0:
        print("\nCould not determine optimal clusters. Using default k=2.")
        best_k = 2
        best_score = 0.0
    else:
        best_idx = np.argmax(silhouette_scores)
        best_k = best_idx + min_clusters
        best_score = raw_scores[best_idx]  # Return the raw score, not the adjusted one

        # Show recommendations based on different metrics
        sil_best_k = np.argmax(raw_scores) + min_clusters
        ch_best_k = np.argmax(ch_scores) + min_clusters

        print("\nCluster recommendations:")
        print(f"- Silhouette score suggests k={sil_best_k}")
        print(f"- Calinski-Harabasz index suggests k={ch_best_k}")
        print(f"- Adjusted score (with higher k preference) suggests k={best_k}")

        # Check if there's disagreement between metrics
        if sil_best_k != ch_best_k:
            print("\nNote: Different metrics suggest different optimal cluster counts.")
            print("This may indicate complex or overlapping cluster structure.")

        # Report on cluster stability
        if best_k > min_clusters and best_k < max_clusters:
            # Check if scores are very close
            near_best_scores = [
                abs(s - silhouette_scores[best_idx]) < 0.02 for s in silhouette_scores
            ]
            if sum(near_best_scores) > 1:
                print(
                    "\nCluster stability warning: Multiple k values have very similar scores."
                )
                print("Consider examining multiple clustering solutions.")

        print(
            f"\nSelected optimal number of clusters: {best_k} (silhouette score: {best_score:.4f})"
        )

    # Ensure correct return types
    # Cast best_k to int, best_score to float, and ensure scores list contains floats
    best_k_int = int(best_k)
    best_score_float = float(best_score if not pd.isna(best_score) else 0.0)
    raw_scores_float = [float(s) if not pd.isna(s) else 0.0 for s in raw_scores]

    return best_k_int, best_score_float, raw_scores_float


def embed_and_cluster(
    X: np.ndarray,
    eps: float,
    min_samples: int,
    nadir_times: Optional[np.ndarray] = None,
    nadir_weight: float = 1.0,
    compare_fudolig: bool = False,
    algorithm: str = "tda_nadir_umap_kmeans_silhouette_k",
    min_clusters: int = 2,
    max_clusters: int = 10,
    favor_higher_k: bool = True,
    penalty_weight: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Return 2D embedding + cluster labels based on selected algorithm.

    Parameters
    ----------
    X : np.ndarray
        Input features for embedding and clustering
    eps : float
        DBSCAN epsilon parameter (only used for DBSCAN-based algorithms)
    min_samples : int
        DBSCAN min_samples parameter (only used for DBSCAN-based algorithms)
    nadir_times : np.ndarray, optional
        Nadir timing values (as % of sleep period)
    nadir_weight : float
        Weight given to nadir timing in clustering
    compare_fudolig : bool
        Whether to also run k=2 clustering for comparison (deprecated)
    algorithm : str
        Clustering algorithm to use:
        - "tda_nadir_umap_kmeans_silhouette_k": K-means clustering with optimal k on UMAP embedding
        - "gmm_paa_hr_bic": Gaussian Mixture Models (GMM) on PAA-HR features with BIC
        - "hac_dtw": Hierarchical Agglomerative Clustering with Dynamic Time Warping
        - "spectral_clustering_paa_hr": Spectral Clustering on PAA-HR features
        - "gmm_slope_bic": Gaussian Mixture Models on HR slope features with BIC
        - "hac_dtw_slope": Hierarchical Agglomerative Clustering with DTW on HR slope
        - "spectral_clustering_slope": Spectral Clustering on HR slope features
    min_clusters : int, default=2
        Minimum number of clusters to try in silhouette analysis
    max_clusters : int, default=10
        Maximum number of clusters to try in silhouette analysis
    favor_higher_k : bool, default=True
        Whether to apply a small penalty to favor higher cluster counts
    penalty_weight : float, default=0.05
        Strength of the penalty for lower k values

    Returns
    -------
    np.ndarray
        2D UMAP embedding array of shape (n_samples, 2) for visualization.
        This is always calculated on the original features for consistent visualization.
    np.ndarray
        Cluster labels based on selected algorithm
    np.ndarray, optional
        K=2 cluster labels for comparison if requested (deprecated)
    """
    try:
        # Ensure we have enough data to process
        if len(X) < min_samples:
            print(
                f"Warning: Not enough data points ({len(X)}) for clustering. Need at least {min_samples}."
            )
            # Return empty arrays with correct types and shapes
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.array([], dtype=np.int32),
                None,
            )

        # Convert input to a standard numpy array and ensure it's 2D
        X_array = np.asarray(X, dtype=np.float32)

        # Handle 1D case (shouldn't happen with our data, but just in case)
        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)

        # Store the original features for UMAP visualization
        X_original = X_array.copy()

        # For derivatives-based methods, we need to preprocess to get the slopes
        if algorithm in ["gmm_slope_bic", "hac_dtw_slope", "spectral_clustering_slope"]:
            # Prepare a list to store the slope arrays
            slope_arrays = []

            # Assuming X contains the HR time series (one per row)
            for i in range(len(X_array)):
                # Calculate slope for each HR time series
                slope = slope_fixed_len(X_array[i], method="resample")
                slope_arrays.append(slope)

            # Convert list of slopes to a 2D array
            X_slope_array = np.vstack(slope_arrays)
            # Use the slope array for further processing
            X_array = X_slope_array

            print(f"Preprocessed HR data to slopes, new shape: {X_array.shape}")

        # Prepare feature matrix, potentially including nadir timing
        if nadir_times is not None and len(nadir_times) == len(X_array):
            # Handle NaN values in nadir times
            valid_nadir = ~np.isnan(nadir_times)
            if not np.any(valid_nadir):
                print(
                    "Warning: All nadir timing values are NaN. Proceeding without nadir timing."
                )
                X_combined = X_array
                X_original_combined = X_original  # For visualization
            else:
                # Fill NaN values with median
                nadir_clean = nadir_times.copy()
                nadir_median = np.nanmedian(nadir_times)
                nadir_clean[~valid_nadir] = nadir_median

                # Normalize nadir times to [0,1] range if we have variation
                nadir_range = np.nanmax(nadir_times) - np.nanmin(nadir_times)
                if nadir_range > 0:
                    nadir_norm = (nadir_clean - np.nanmin(nadir_times)) / nadir_range
                else:
                    # Handle the case where all values are identical
                    nadir_norm = np.zeros_like(nadir_clean)

                # Weight and combine with other features
                X_combined = np.column_stack([X_array, nadir_norm * nadir_weight])
                # Also combine original features with nadir timing for visualization
                X_original_combined = np.column_stack(
                    [X_original, nadir_norm * nadir_weight]
                )

                print(
                    f"Combined feature matrix shape with nadir timing: {X_combined.shape}"
                )
        else:  # This else aligns with the outer if
            X_combined = X_array
            X_original_combined = X_original  # For visualization

        # Create UMAP embedding for visualization - ALWAYS use original features for consistent visualization
        n_neighbors = min(
            30, max(2, len(X_original_combined) - 1)
        )  # Ensure n_neighbors is valid (at least 2)

        print("Creating UMAP embedding for visualization from original features")
        umap_reducer = umap.UMAP(
            n_neighbors=n_neighbors, min_dist=0.1, metric="euclidean", random_state=42
        )

        # Get the 2D embedding from original features for visualization
        embedding_result = umap_reducer.fit_transform(X_original_combined)
        umap_embedding = np.asarray(embedding_result, dtype=np.float32)

        # Initialize variables for clustering results
        cluster_labels = None
        k2_labels = None

        # Now perform the actual clustering using the appropriate features and algorithm
        if algorithm == "tda_nadir_umap_kmeans_silhouette_k":
            # In this case we do use the UMAP embedding for clustering
            print(
                "Finding optimal number of clusters using Silhouette Analysis on UMAP embedding"
            )
            optimal_k, best_score, _ = find_optimal_clusters(
                umap_embedding,
                min_clusters=min_clusters,
                max_clusters=max_clusters,
                favor_higher_k=favor_higher_k,
                penalty_weight=penalty_weight,
            )
            print(f"Applying KMeans with optimal k={optimal_k} on UMAP embedding")
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(umap_embedding)

        elif algorithm == "gmm_paa_hr_bic":
            # Apply Gaussian Mixture Model directly to PAA-HR features
            print("Applying Gaussian Mixture Model (GMM) to PAA-HR features with BIC")

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_array)

            # Find optimal number of components using BIC
            n_components_range = range(min_clusters, max_clusters + 1)
            bic_values = []
            gmm_models = []

            for n_components in n_components_range:
                # Initialize and fit GMM
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type="full",  # Allow for different cluster shapes
                    random_state=42,
                    n_init=5,  # Multiple initializations to avoid local optima
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                    gmm.fit(X_scaled)

                bic_values.append(gmm.bic(X_scaled))
                gmm_models.append(gmm)

            # Find optimal number of components (minimum BIC)
            bic_values = np.array(bic_values)
            best_idx = np.argmin(bic_values)
            optimal_n = n_components_range[best_idx]

            print(f"Optimal number of GMM components based on BIC: {optimal_n}")

            # Get best GMM model and predict clusters
            best_gmm = gmm_models[best_idx]
            cluster_labels = best_gmm.predict(X_scaled)

            # Plot BIC vs number of components
            plt.figure(figsize=(10, 6))
            plt.plot(n_components_range, bic_values, "o-")
            plt.axvline(
                x=optimal_n,
                color="r",
                linestyle="--",
                label=f"Optimal n_components = {optimal_n}",
            )
            plt.xlabel("Number of components")
            plt.ylabel("BIC score")
            plt.title("BIC Scores for GMM Components")
            plt.legend()
            plt.savefig("gmm_bic_score.png")
            plt.close()

        elif algorithm == "hac_dtw":
            # Apply Hierarchical Agglomerative Clustering with DTW distance
            print("Applying Hierarchical Agglomerative Clustering with DTW")

            # Calculate pairwise DTW distances
            print("Computing DTW distance matrix...")
            n_samples = len(X_array)
            dtw_matrix = np.zeros((n_samples, n_samples))

            # Compute upper triangular part of the matrix (symmetric)
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    distance = dtw(X_array[i], X_array[j])
                    dtw_matrix[i, j] = distance
                    dtw_matrix[j, i] = distance

            # Find optimal number of clusters using silhouette score
            silhouette_scores = []
            labels_list = []

            for n_clusters in range(min_clusters, max_clusters + 1):
                # Apply HAC with precomputed distances
                hac = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric="precomputed",
                    linkage="average",  # 'average' often works well with DTW
                )
                labels = hac.fit_predict(dtw_matrix)
                labels_list.append(labels)

                # Skip silhouette score if we have only one cluster
                if len(np.unique(labels)) < 2:
                    silhouette_scores.append(-1)  # Invalid score
                    continue

                # Compute silhouette score using the precomputed distances
                score = silhouette_score(dtw_matrix, labels, metric="precomputed")
                silhouette_scores.append(score)

            # Find optimal number of clusters
            if max(silhouette_scores) > 0:
                best_idx = np.argmax(silhouette_scores)
                optimal_n = best_idx + min_clusters
                print(
                    f"Optimal number of HAC clusters based on silhouette: {optimal_n}"
                )
                cluster_labels = labels_list[best_idx]
            else:
                # Default to min_clusters if no valid scores
                print(
                    f"No valid silhouette scores, defaulting to {min_clusters} clusters"
                )
                optimal_n = min_clusters
                hac = AgglomerativeClustering(
                    n_clusters=optimal_n, metric="precomputed", linkage="average"
                )
                cluster_labels = hac.fit_predict(dtw_matrix)

        elif algorithm == "spectral_clustering_paa_hr":
            # Apply Spectral Clustering on PAA-HR features
            print("Applying Spectral Clustering on PAA-HR features")

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_array)

            # Construct similarity graph using RBF kernel
            # Calculate pairwise Euclidean distances
            distances = pdist(X_scaled, metric="euclidean")
            dist_matrix = squareform(distances)

            # Convert to similarity using Gaussian kernel
            gamma = 1.0 / X_scaled.shape[1]  # Default gamma value
            affinity_matrix = np.exp(-gamma * (dist_matrix**2))

            # Find optimal number of clusters using eigengap heuristic
            from scipy.sparse.linalg import eigsh
            from scipy.linalg import eigh

            # Calculate the normalized graph Laplacian
            n_samples = X_scaled.shape[0]
            degree_matrix = np.diag(np.sum(affinity_matrix, axis=1))
            laplacian = degree_matrix - affinity_matrix
            normed_laplacian = np.eye(n_samples) - np.linalg.inv(
                np.sqrt(degree_matrix)
            ) @ affinity_matrix @ np.linalg.inv(np.sqrt(degree_matrix))

            # Get eigenvalues
            try:
                if n_samples > 100:
                    # Use sparse method for large matrices
                    eigenvalues, _ = eigsh(
                        normed_laplacian, k=max_clusters + 1, which="SM"
                    )
                else:
                    # Use dense method for smaller matrices
                    eigenvalues, _ = eigh(normed_laplacian)
                    eigenvalues = eigenvalues[: max_clusters + 1]

                # Sort eigenvalues
                eigenvalues = np.sort(eigenvalues)

                # Calculate eigengaps
                eigengaps = np.diff(eigenvalues)

                # Find optimal k (add 1 because diff reduces length by 1)
                # Look for large gaps in the smallest eigenvalues
                optimal_k = np.argmax(eigengaps[: max_clusters - 1]) + 1

                # Ensure optimal_k is in the valid range
                optimal_k = max(min_clusters, min(optimal_k, max_clusters))
            except Exception as e:
                print(f"Error in eigengap calculation: {e}")
                print(f"Defaulting to {min_clusters} clusters")
                optimal_k = min_clusters

            print(
                f"Optimal number of clusters based on eigengap heuristic: {optimal_k}"
            )

            # Apply Spectral Clustering with the determined k
            spectral = SpectralClustering(
                n_clusters=optimal_k,
                affinity="precomputed",
                random_state=42,
                assign_labels="kmeans",
            )

            cluster_labels = spectral.fit_predict(affinity_matrix)

        elif algorithm == "gmm_slope_bic":
            # Apply Gaussian Mixture Model to HR slope features
            print("Applying Gaussian Mixture Model (GMM) to HR slope features with BIC")

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_array)

            # Find optimal number of components using BIC
            n_components_range = range(min_clusters, max_clusters + 1)
            bic_values = []
            gmm_models = []

            for n_components in n_components_range:
                # Initialize and fit GMM
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type="full",  # Allow for different cluster shapes
                    random_state=42,
                    n_init=5,  # Multiple initializations to avoid local optima
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                    gmm.fit(X_scaled)

                bic_values.append(gmm.bic(X_scaled))
                gmm_models.append(gmm)

            # Find optimal number of components (minimum BIC)
            bic_values = np.array(bic_values)
            best_idx = np.argmin(bic_values)
            optimal_n = n_components_range[best_idx]

            print(f"Optimal number of GMM components based on BIC: {optimal_n}")

            # Get best GMM model and predict clusters
            best_gmm = gmm_models[best_idx]
            cluster_labels = best_gmm.predict(X_scaled)

            # Plot BIC vs number of components
            plt.figure(figsize=(10, 6))
            plt.plot(n_components_range, bic_values, "o-")
            plt.axvline(
                x=optimal_n,
                color="r",
                linestyle="--",
                label=f"Optimal n_components = {optimal_n}",
            )
            plt.xlabel("Number of components")
            plt.ylabel("BIC score")
            plt.title("BIC Scores for GMM Components (Slope Features)")
            plt.legend()
            plt.savefig("gmm_slope_bic_score.png")
            plt.close()

        elif algorithm == "hac_dtw_slope":
            # Apply Hierarchical Agglomerative Clustering with DTW on HR slope
            print("Applying Hierarchical Agglomerative Clustering with DTW on HR slope")

            # Calculate pairwise DTW distances on slope features
            print("Computing DTW distance matrix for slope features...")
            n_samples = len(X_array)
            dtw_matrix = np.zeros((n_samples, n_samples))

            # Compute upper triangular part of the matrix (symmetric)
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    distance = dtw(X_array[i], X_array[j])
                    dtw_matrix[i, j] = distance
                    dtw_matrix[j, i] = distance

            # Find optimal number of clusters using silhouette score
            silhouette_scores = []
            labels_list = []

            for n_clusters in range(min_clusters, max_clusters + 1):
                # Apply HAC with precomputed distances
                hac = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric="precomputed",
                    linkage="average",  # 'average' often works well with DTW
                )
                labels = hac.fit_predict(dtw_matrix)
                labels_list.append(labels)

                # Skip silhouette score if we have only one cluster
                if len(np.unique(labels)) < 2:
                    silhouette_scores.append(-1)  # Invalid score
                    continue

                # Compute silhouette score using the precomputed distances
                score = silhouette_score(dtw_matrix, labels, metric="precomputed")
                silhouette_scores.append(score)

            # Find optimal number of clusters
            if max(silhouette_scores) > 0:
                best_idx = np.argmax(silhouette_scores)
                optimal_n = best_idx + min_clusters
                print(
                    f"Optimal number of HAC clusters based on silhouette: {optimal_n}"
                )
                cluster_labels = labels_list[best_idx]
            else:
                # Default to min_clusters if no valid scores
                print(
                    f"No valid silhouette scores, defaulting to {min_clusters} clusters"
                )
                optimal_n = min_clusters
                hac = AgglomerativeClustering(
                    n_clusters=optimal_n, metric="precomputed", linkage="average"
                )
                cluster_labels = hac.fit_predict(dtw_matrix)

        elif algorithm == "spectral_clustering_slope":
            # Apply Spectral Clustering on HR slope features
            print("Applying Spectral Clustering on HR slope features")

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_array)

            # Construct similarity graph using RBF kernel
            # Calculate pairwise Euclidean distances
            distances = pdist(X_scaled, metric="euclidean")
            dist_matrix = squareform(distances)

            # Convert to similarity using Gaussian kernel
            gamma = 1.0 / X_scaled.shape[1]  # Default gamma value
            affinity_matrix = np.exp(-gamma * (dist_matrix**2))

            # Find optimal number of clusters using eigengap heuristic
            from scipy.sparse.linalg import eigsh
            from scipy.linalg import eigh

            # Calculate the normalized graph Laplacian
            n_samples = X_scaled.shape[0]
            degree_matrix = np.diag(np.sum(affinity_matrix, axis=1))
            laplacian = degree_matrix - affinity_matrix
            normed_laplacian = np.eye(n_samples) - np.linalg.inv(
                np.sqrt(degree_matrix)
            ) @ affinity_matrix @ np.linalg.inv(np.sqrt(degree_matrix))

            # Get eigenvalues
            try:
                if n_samples > 100:
                    # Use sparse method for large matrices
                    eigenvalues, _ = eigsh(
                        normed_laplacian, k=max_clusters + 1, which="SM"
                    )
                else:
                    # Use dense method for smaller matrices
                    eigenvalues, _ = eigh(normed_laplacian)
                    eigenvalues = eigenvalues[: max_clusters + 1]

                # Sort eigenvalues
                eigenvalues = np.sort(eigenvalues)

                # Calculate eigengaps
                eigengaps = np.diff(eigenvalues)

                # Find optimal k (add 1 because diff reduces length by 1)
                # Look for large gaps in the smallest eigenvalues
                optimal_k = np.argmax(eigengaps[: max_clusters - 1]) + 1

                # Ensure optimal_k is in the valid range
                optimal_k = max(min_clusters, min(optimal_k, max_clusters))
            except Exception as e:
                print(f"Error in eigengap calculation: {e}")
                print(f"Defaulting to {min_clusters} clusters")
                optimal_k = min_clusters

            print(
                f"Optimal number of clusters based on eigengap heuristic: {optimal_k}"
            )

            # Apply Spectral Clustering with the determined k
            spectral = SpectralClustering(
                n_clusters=optimal_k,
                affinity="precomputed",
                random_state=42,
                assign_labels="kmeans",
            )

            cluster_labels = spectral.fit_predict(affinity_matrix)

        # Fixed k=3 versions of all algorithms
        elif algorithm == "tda_nadir_umap_kmeans_3":
            # K-means with fixed k=3 on UMAP embedding
            print("Applying K-means with fixed k=3 on UMAP embedding")
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(umap_embedding)

        elif algorithm == "gmm_paa_hr_3":
            # GMM with fixed k=3 on original features
            print(
                "Applying Gaussian Mixture Model (GMM) with fixed k=3 on PAA-HR features"
            )

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_array)

            # Apply GMM with k=3
            gmm = GaussianMixture(
                n_components=3,
                covariance_type="full",
                random_state=42,
                n_init=5,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                gmm.fit(X_scaled)

            cluster_labels = gmm.predict(X_scaled)

        elif algorithm == "spectral_clustering_paa_hr_3":
            # Spectral Clustering with fixed k=3
            print("Applying Spectral Clustering with fixed k=3 on PAA-HR features")

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_array)

            # Calculate pairwise distances
            distances = pdist(X_scaled, metric="euclidean")
            dist_matrix = squareform(distances)

            # Convert to similarity using Gaussian kernel
            gamma = 1.0 / X_scaled.shape[1]
            affinity_matrix = np.exp(-gamma * (dist_matrix**2))

            # Apply Spectral Clustering with k=3
            spectral = SpectralClustering(
                n_clusters=3,
                affinity="precomputed",
                random_state=42,
                assign_labels="kmeans",
            )

            cluster_labels = spectral.fit_predict(affinity_matrix)

        elif algorithm == "hac_dtw_3":
            # Hierarchical Agglomerative Clustering with DTW and fixed k=3
            print("Applying Hierarchical Clustering with DTW and fixed k=3")

            # Calculate pairwise DTW distances
            print("Computing DTW distance matrix...")
            n_samples = len(X_array)
            dtw_matrix = np.zeros((n_samples, n_samples))

            # Compute upper triangular part of the matrix (symmetric)
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    distance = dtw(X_array[i], X_array[j])
                    dtw_matrix[i, j] = distance
                    dtw_matrix[j, i] = distance

            # Apply HAC with k=3
            hac = AgglomerativeClustering(
                n_clusters=3,
                metric="precomputed",
                linkage="average",
            )

            cluster_labels = hac.fit_predict(dtw_matrix)

        elif algorithm == "gmm_slope_3":
            # GMM on slope features with fixed k=3
            print("Applying GMM on HR slope features with fixed k=3")

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_array)

            # Apply GMM with k=3
            gmm = GaussianMixture(
                n_components=3,
                covariance_type="full",
                random_state=42,
                n_init=5,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                gmm.fit(X_scaled)

            cluster_labels = gmm.predict(X_scaled)

        elif algorithm == "hac_dtw_slope_3":
            # Hierarchical Agglomerative Clustering with DTW on slope features with fixed k=3
            print(
                "Applying Hierarchical Clustering with DTW on HR slope features with fixed k=3"
            )

            # Calculate pairwise DTW distances
            print("Computing DTW distance matrix...")
            n_samples = len(X_array)
            dtw_matrix = np.zeros((n_samples, n_samples))

            # Compute upper triangular part of the matrix (symmetric)
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    distance = dtw(X_array[i], X_array[j])
                    dtw_matrix[i, j] = distance
                    dtw_matrix[j, i] = distance

            # Apply HAC with k=3
            hac = AgglomerativeClustering(
                n_clusters=3,
                metric="precomputed",
                linkage="average",
            )

            cluster_labels = hac.fit_predict(dtw_matrix)

        elif algorithm == "spectral_clustering_slope_3":
            # Spectral Clustering on slope features with fixed k=3
            print("Applying Spectral Clustering on HR slope features with fixed k=3")

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_array)

            # Calculate pairwise distances
            distances = pdist(X_scaled, metric="euclidean")
            dist_matrix = squareform(distances)

            # Convert to similarity using Gaussian kernel
            gamma = 1.0 / X_scaled.shape[1]
            affinity_matrix = np.exp(-gamma * (dist_matrix**2))

            # Apply Spectral Clustering with k=3
            spectral = SpectralClustering(
                n_clusters=3,
                affinity="precomputed",
                random_state=42,
                assign_labels="kmeans",
            )

            cluster_labels = spectral.fit_predict(affinity_matrix)

        else:
            raise ValueError(f"Unknown clustering algorithm: {algorithm}")

        # Ensure output types match our function signature
        cluster_labels = np.asarray(cluster_labels, dtype=np.int32)

        return umap_embedding, cluster_labels, k2_labels
    except Exception as e:
        print(f"Error in embedding and clustering: {e}")
        print(f"Input shape: {X.shape if hasattr(X, 'shape') else 'unknown'}")
        # Return empty arrays with correct shapes
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.array([], dtype=np.int32),
            None,
        )


def meta_cluster_by_centroid(eps_df: pd.DataFrame, n_meta: int) -> pd.Series:
    """Hierarchical meta-clustering on average slope centroids.

    Parameters
    ----------
    eps_df : pd.DataFrame
        DataFrame with episodes and 'cluster' and 'slope' columns
    n_meta : int
        Number of meta-clusters to create

    Returns
    -------
    pd.Series
        Series mapping original cluster IDs to meta-cluster IDs
    """
    # Get unique clusters excluding noise points (-1)
    unique_clusters = eps_df[eps_df["cluster"] != -1]["cluster"].unique()
    n_unique_clusters = len(unique_clusters)

    if n_unique_clusters < 2:
        print(
            f"Warning: Only {n_unique_clusters} clusters found (excluding noise). Cannot perform meta-clustering."
        )
        # Return original cluster labels
        return eps_df["cluster"]

    # Adjust n_meta if necessary
    if n_meta > n_unique_clusters:
        print(
            f"Warning: Requested {n_meta} meta-clusters but only {n_unique_clusters} clusters available."
        )
        print(f"Adjusting to use {n_unique_clusters} meta-clusters instead.")
        n_meta = n_unique_clusters

    # Continue with meta-clustering
    slopes = eps_df.groupby("cluster")["slope"].apply(
        lambda arrs: np.mean(np.stack(list(arrs.values)), axis=0)
    )
    centroids = np.stack(list(slopes.values))

    print(
        f"Performing meta-clustering with {n_meta} clusters on {len(centroids)} original clusters"
    )

    agg = AgglomerativeClustering(n_clusters=n_meta).fit(centroids)
    # map original cluster -> meta label
    mapping = dict(zip(slopes.index.tolist(), agg.labels_))
    return eps_df["cluster"].map(mapping)


def calculate_patient_cluster_consistency(
    eps_df: pd.DataFrame, use_meta: bool = False
) -> pd.DataFrame:
    """
    Calculate metrics of patient-level cluster consistency and aggregate patient characteristics.

    This function calculates how consistently each patient's sleep episodes are assigned
    to the same cluster, using either entropy or modal proportion metrics.

    Parameters
    ----------
    eps_df : pd.DataFrame
        DataFrame with cluster assignments and episode characteristics
    use_meta : bool
        Whether to use meta-clusters instead of regular clusters

    Returns
    -------
    pd.DataFrame
        Patient-level DataFrame with consistency metrics and aggregated characteristics
    """
    cluster_col = (
        "meta_cluster" if use_meta and "meta_cluster" in eps_df.columns else "cluster"
    )
    print(f"Calculating patient-level {cluster_col} consistency")

    # Filter out noise points (-1) from cluster calculations
    df_valid = eps_df[eps_df[cluster_col] != -1].copy()

    # Initialize patient-level results
    patient_results = []

    # Group by patient ID
    for patient_id, patient_df in df_valid.groupby("id"):
        # Skip patients with fewer than 4 episodes (changed from 2)
        if len(patient_df) < 3:
            print(
                f"  Skipping patient {patient_id} for consistency calculation (only {len(patient_df)} valid episodes, need >= 3)."
            )
            continue

        # Calculate cluster distribution
        cluster_counts = patient_df[cluster_col].value_counts()
        cluster_probs = cluster_counts / len(patient_df)

        # Calculate modal cluster and its proportion
        modal_cluster = cluster_counts.idxmax()
        modal_proportion = cluster_counts.max() / len(patient_df)

        # Calculate cluster entropy
        entropy = -np.sum(cluster_probs * np.log2(cluster_probs))
        # Normalize by maximum possible entropy for the number of episodes
        max_entropy = np.log2(min(len(cluster_counts), len(patient_df)))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        # Calculate mean sleep characteristics
        patient_metrics = {
            "id": patient_id,
            "n_episodes": len(patient_df),
            "n_clusters": len(cluster_counts),
            "modal_cluster": modal_cluster,
            "modal_proportion": modal_proportion,
            "cluster_entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "cluster_consistency": 1 - normalized_entropy,  # Higher = more consistent
        }

        # Aggregate sleep characteristics
        continuous_vars = [
            "se",
            "waso",
            "duration_secs",
            "episode_hr_mean",
            "nadir_time_pct",
            "nadir_hr",
            "abs_chronotype_desync",
            "chronotype_desync",
        ]
        for var in continuous_vars:
            if var in patient_df.columns:
                patient_metrics[f"mean_{var}"] = patient_df[var].mean()
                patient_metrics[f"std_{var}"] = patient_df[var].std()
                patient_metrics[f"cv_{var}"] = (
                    (patient_df[var].std() / patient_df[var].mean())
                    if patient_df[var].mean() != 0
                    else np.nan
                )

        # Add patient-level characteristics (same for all episodes)
        static_vars = ["age", "sex", "resting_hr", "sri", "patient_chronotype"]
        for var in static_vars:
            if var in patient_df.columns:
                patient_metrics[var] = patient_df[var].iloc[0]

        patient_results.append(patient_metrics)

    # Create DataFrame and handle potential empty results
    if not patient_results:
        print(
            "Warning: No patients with multiple episodes found for consistency analysis"
        )
        # Return empty DataFrame with expected columns
        return pd.DataFrame(
            columns=[
                "id",
                "n_episodes",
                "n_clusters",
                "modal_cluster",
                "modal_proportion",
                "cluster_entropy",
                "normalized_entropy",
                "cluster_consistency",
            ]
        )

    patient_df = pd.DataFrame(patient_results)

    # Calculate correlations between consistency metrics and sleep characteristics
    print(f"\nSpearman correlations with cluster consistency (1-normalized entropy):")
    correlation_cols = [c for c in patient_df.columns if c.startswith("mean_")] + [
        "age",
        "patient_chronotype",
    ]
    correlation_cols = [c for c in correlation_cols if c in patient_df.columns]

    if correlation_cols:
        correlations = []
        for col in correlation_cols:
            # Ensure both columns have variance before attempting correlation
            if (
                patient_df["cluster_consistency"].nunique() > 1
                and patient_df[col].nunique() > 1
            ):
                try:
                    corr_result = stats.spearmanr(
                        patient_df["cluster_consistency"],
                        patient_df[col],
                        nan_policy="omit",
                    )
                    # Extract correlation and p-value using safer methods
                    # Check for named tuple attributes first (newer scipy)
                    if hasattr(corr_result, "correlation") and hasattr(
                        corr_result, "pvalue"
                    ):
                        corr = float(corr_result.correlation)
                        p = float(corr_result.pvalue)
                        # Handle potential NaN results from spearmanr
                        if pd.isna(corr) or pd.isna(p):
                            print(
                                f"Warning: spearmanr returned NaN for {col}. Skipping."
                            )
                            continue

                    # Fallback to tuple indexing (older scipy)
                    elif isinstance(corr_result, tuple) and len(corr_result) >= 2:
                        # Convert each element to float individually, handle potential errors
                        try:
                            corr = float(corr_result[0])
                            p = float(corr_result[1])
                            if pd.isna(corr) or pd.isna(p):
                                print(
                                    f"Warning: spearmanr returned NaN (tuple access) for {col}. Skipping."
                                )
                                continue
                        except (TypeError, ValueError):
                            print(
                                f"Could not convert spearmanr tuple results to float for {col}. Skipping."
                            )
                            continue
                    else:
                        # If we can\'t extract values in either format, skip
                        print(
                            f"Could not extract correlation values for {col} (Unexpected format: {type(corr_result)}). Skipping."
                        )
                        continue

                    correlations.append(
                        {
                            "variable": col,
                            "correlation": corr,
                            "p_value": p,
                            "significant": p
                            < 0.05,  # Check significance after successful extraction
                        }
                    )
                except ValueError as e:
                    # Catch potential errors during spearmanr calculation itself (e.g., insufficient data after omit)
                    print(f"ValueError calculating spearmanr for {col}: {e}. Skipping.")
                    continue

            else:  # Skip if one of the columns lacks variance
                print(
                    f"Skipping correlation for {col}: Insufficient variance in the data."
                )

        if correlations:
            corr_df = pd.DataFrame(correlations)
            print(corr_df.sort_values("p_value").to_string(index=False))

    # Fit a simple OLS model to predict consistency
    try:
        import statsmodels.formula.api as smf

        # Select potential predictors (exclude ID, cluster metrics, etc.)
        predictors = [
            c
            for c in correlation_cols
            if c in patient_df.columns
            and patient_df[c].nunique() > 1
            and not pd.isna(patient_df[c]).all()
        ]

        if len(predictors) > 0 and len(patient_df) > len(predictors) + 2:
            formula_parts = []
            for pred in predictors:
                # Handle column names with special characters
                if pred == "sex":
                    # For categorical variables, might need to handle differently
                    formula_parts.append(f"C({pred})")
                else:
                    formula_parts.append(pred)

            formula = "cluster_consistency ~ " + " + ".join(formula_parts)
            model = smf.ols(formula, data=patient_df).fit()
            print("\nOLS Model predicting cluster consistency:")
            print(
                f"R-squared: {model.rsquared:.3f}, Adj R-squared: {model.rsquared_adj:.3f}"
            )
            print("Significant predictors (p<0.05):")

            # Extract significant predictors
            for var, p_value in model.pvalues.items():
                if p_value < 0.05 and var != "Intercept":
                    coef = model.params[var]
                    print(f"  {var}: coef={coef:.4f}, p={p_value:.4f}")
    except Exception as e:
        print(f"Could not fit OLS model: {e}")

    return patient_df
