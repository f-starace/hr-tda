"""
Visualization module for sleep data analysis.

This module provides functions for plotting heart rate patterns,
cluster visualizations, and statistical results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.colors as mcolors

# Constants
TARGET_LEN = 100  # Standard length for resampled heart rate series


def _format_p_value_for_title(p_value: float) -> str:
    """Formats a p-value for display in a plot title."""
    if p_value < 0.0001 and p_value != 0:
        return "$p < 10^{-4}$"
    else:
        return f"$p = {p_value:.4f}$"


def plot_cluster_hr(
    hr_data_by_cluster: Dict[int, np.ndarray],
    ax: Optional[Axes] = None,
    out_path: Optional[Union[str, Path]] = None,
):
    """
    Plot average heart rate curves for each cluster with confidence intervals.

    Parameters
    ----------
    hr_data_by_cluster : Dict[int, np.ndarray]
        Dictionary mapping cluster IDs to arrays of HR curves (shape: n_episodes x n_timepoints)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.
    out_path : str or Path, optional
        Path to save the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Use a safer way to get colors
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    x = np.arange(TARGET_LEN)

    # For statistical testing
    all_hr_data = []
    cluster_labels = []

    # Plot each cluster
    for i, (cluster_id, hr_curves) in enumerate(hr_data_by_cluster.items()):
        color = color_cycle[i % len(color_cycle)]

        # Make sure hr_curves is the right shape
        hr_curves = np.array(hr_curves)

        # Calculate mean
        mean_hr = np.mean(hr_curves, axis=0)

        # Ensure all arrays are 1D
        mean_hr = np.asarray(mean_hr).flatten()

        # Plot mean line - Use just the cluster ID number without the "Cluster" prefix
        ax.plot(x, mean_hr, label=f"{cluster_id}", color=color, linewidth=2)

        # Only add confidence interval if we have multiple curves
        if len(hr_curves) > 1:
            # Calculate SEM and CI
            sem_hr = stats.sem(hr_curves, axis=0)
            sem_hr = np.asarray(sem_hr).flatten()
            ci95_hr = sem_hr * 1.96  # 95% confidence interval

            # Make sure dimensions match
            if len(x) == len(mean_hr) == len(ci95_hr):
                ax.fill_between(
                    x, mean_hr - ci95_hr, mean_hr + ci95_hr, color=color, alpha=0.2
                )

        # Save data for statistical testing
        all_hr_data.append(hr_curves)
        cluster_labels.extend([cluster_id] * len(hr_curves))

    ax.set_xlabel("Normalized Time (% of sleep episode)")
    ax.set_ylabel("Heart Rate (BPM)")

    # Set legend to be horizontal at the top
    ax.legend(
        title="Cluster",
        ncol=min(8, len(hr_data_by_cluster)),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        frameon=False,
    )

    sns.despine()

    # Save the figure if out_path is provided
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")

    return ax


def plot_cluster_slope(
    avg_slope: Dict[int, np.ndarray],
    ax: Optional[Axes] = None,
    out_path: Optional[Union[str, Path]] = None,
):
    """
    Plot average heart rate slope by cluster.

    Parameters
    ----------
    avg_slope : Dict[int, np.ndarray]
        Dictionary mapping cluster IDs to average slope arrays
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.
    out_path : str or Path, optional
        Path to save the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    for cl, slope in avg_slope.items():
        ax.plot(slope, label=f"C{cl}")

    ax.set_xlabel("Resampled time (bins)")
    ax.set_ylabel("HR slope (bpm/bin)")
    ax.legend(title="Cluster")
    ax.set_title("Average HR-slope shape per cluster")
    sns.despine()

    # Save the figure if out_path is provided
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")

    return ax


def plot_umap(
    df: pd.DataFrame,
    ax: Optional[Axes] = None,
    hue_col: str = "cluster",
    out_path: Optional[Union[str, Path]] = None,
):
    """
    Plot UMAP embedding colored by cluster.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing umap1 and umap2 columns and the hue column
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.
    hue_col : str, default="cluster"
        Column to use for coloring points
    out_path : str or Path, optional
        Path to save the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    sns.scatterplot(
        data=df,
        x="umap1",
        y="umap2",
        hue=hue_col,
        palette="tab10",
        ax=ax,
        alpha=0.7,
        s=30,
    )
    ax.set_title(f"UMAP embedding of HR-slope TDA vectors (colored by {hue_col})")
    sns.despine()

    # Save the figure if out_path is provided
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")

    return ax


def plot_hr_summary_by_cluster(
    stats_df: pd.DataFrame,
    ax: Optional[Axes] = None,
    out_path: Optional[Union[str, Path]] = None,
):
    """
    Create a summary box plot of heart rate statistics by cluster.

    Parameters
    ----------
    stats_df : pd.DataFrame
        DataFrame with HR statistics by cluster (must contain cluster, mean_hr, min_hr, and max_hr columns)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    out_path : str or Path, optional
        Path to save the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data for box plot
    clusters = stats_df["cluster"].tolist()
    means = stats_df["mean_hr"].tolist()
    mins = stats_df["min_hr"].tolist()
    maxs = stats_df["max_hr"].tolist()

    # Set positions for each cluster
    positions = np.arange(len(clusters))
    width = 0.5

    # Use standard colors
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Create box plot elements manually for more control
    for i, (cluster, mean_val, min_val, max_val) in enumerate(
        zip(clusters, means, mins, maxs)
    ):
        color = color_cycle[i % len(color_cycle)]

        # Plot min-max range - Use just the cluster ID without "Cluster" prefix
        ax.bar(
            positions[i],
            max_val - min_val,
            bottom=min_val,
            width=width,
            color=color,
            alpha=0.3,
            label=f"{cluster}",
        )

        # Plot mean as a line
        ax.hlines(
            mean_val,
            positions[i] - width / 2,
            positions[i] + width / 2,
            color=color,
            linewidth=2,
        )

    # Customize plot
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{c}" for c in clusters])
    ax.set_ylabel("Heart Rate (BPM)")
    ax.set_title("Heart Rate Summary by Cluster")

    # Create custom legend with horizontal layout
    legend_elements = [
        Patch(facecolor="gray", alpha=0.3, label="Min-Max Range"),
        Line2D([0], [0], color="black", linewidth=2, label="Mean HR"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        frameon=False,
    )

    sns.despine()

    # Save the figure if out_path is provided
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")

    return ax


def analyze_cluster_characteristics(
    df: pd.DataFrame, out_dir: Optional[Union[str, Path]] = None, use_meta: bool = False
) -> pd.DataFrame:
    """
    Analyze demographic and sleep characteristics by cluster and create visualizations.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with cluster assignments and patient characteristics
    out_dir : str or Path, optional
        Output directory for saving plots and statistics
    use_meta : bool, default=False
        Whether to use meta-clusters instead of regular clusters

    Returns
    -------
    pd.DataFrame
        Summary statistics by cluster
    """
    # Determine which cluster variable to use
    cluster_col = (
        "meta_cluster" if use_meta and "meta_cluster" in df.columns else "cluster"
    )
    cluster_type = "Meta-cluster" if use_meta else "Cluster"

    # Create clusters directory if it doesn't exist
    clusters_dir = None
    if out_dir is not None:
        out_dir = Path(out_dir)
        clusters_dir = out_dir / ("meta_clusters" if use_meta else "clusters")
        clusters_dir.mkdir(exist_ok=True)

    # Variables to analyze - include nadir timing variables if available
    variables = [
        "sex",
        "age",
        "se",
        "waso",
        "duration_secs",
        "patient_chronotype",
        "chronotype_desync",
        "episode_hr_mean",
        "sfi",
    ]

    # Add nadir timing variables if available
    nadir_vars = [
        "nadir_time_pct",
        "nadir_time_hours",
        "nadir_hr",
        "time_to_nadir_normalized",
    ]

    for var in nadir_vars:
        if var in df.columns:
            variables.append(var)

    # Filter to only include columns that exist
    variables = [var for var in variables if var in df.columns]

    # Create summary by cluster
    summary_stats = []

    # For each cluster
    for cluster_id, cluster_df in df.groupby(cluster_col):
        if cluster_id == -1:  # Skip noise points
            continue

        # Basic stats
        n_episodes = len(cluster_df)
        n_patients = cluster_df["id"].nunique()

        cluster_stats = {
            cluster_col: cluster_id,
            "n_episodes": n_episodes,
            "n_patients": n_patients,
        }

        # Add statistics for continuous variables
        for var in variables:
            if var == "sex":
                # Special handling for categorical variable
                sex_counts = cluster_df["sex"].value_counts(normalize=True) * 100
                male_pct = sex_counts.get("M", 0)
                cluster_stats["pct_male"] = male_pct
            else:
                # Continuous variables
                if var in cluster_df.columns:
                    cluster_stats[f"{var}_mean"] = cluster_df[var].mean()
                    cluster_stats[f"{var}_std"] = cluster_df[var].std()

        summary_stats.append(cluster_stats)

        # Create individual cluster HR plot if output directory is provided
        if clusters_dir is not None and "hr_eq" in cluster_df.columns:
            hr_curves = np.stack(list(cluster_df["hr_eq"].values))

            # Create a grid of 9 example curves (or fewer if there aren't enough)
            n_examples = min(9, len(hr_curves))
            n_cols = 3
            n_rows = (n_examples + n_cols - 1) // n_cols  # Ceiling division

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8))

            # Calculate the mean curve
            mean_curve = np.mean(hr_curves, axis=0)

            # Flatten axes array for easier indexing
            axes_flat = axes.flatten() if n_rows * n_cols > 1 else [axes]

            # Instead of regular interval selection, find curves most similar to mean
            # Calculate MSE between each curve and the mean curve
            mse_values = np.mean((hr_curves - mean_curve) ** 2, axis=1)

            # Get indices of curves with lowest MSE (most similar to mean)
            indices = np.argsort(mse_values)[:n_examples]

            x_values = np.arange(len(hr_curves[0]))

            # Plot each example
            for i in range(n_examples):
                curve = hr_curves[indices[i]]
                axes_flat[i].plot(x_values, curve, color="black")
                axes_flat[i].plot(
                    x_values, mean_curve, color="red", linestyle="--", alpha=0.7
                )
                # Add MSE value to title
                mse = mse_values[indices[i]]
                axes_flat[i].set_title(f"Rep. Example {i+1} (MSE: {mse:.2f})")

                # Set y-axis range from 55 to 75
                axes_flat[i].set_ylim(55, 75)

                # Mark the nadir if available
                if "nadir_time_pct" in cluster_df.columns:
                    nadir_pct = cluster_df["nadir_time_pct"].iloc[indices[i]]
                    if not np.isnan(nadir_pct):
                        nadir_pos = int(nadir_pct * len(curve))
                        nadir_pos = min(
                            max(0, nadir_pos), len(curve) - 1
                        )  # Ensure in bounds
                        nadir_val = curve[nadir_pos]
                        axes_flat[i].plot(nadir_pos, nadir_val, "ko", markersize=6)

            # Turn off any unused subplots
            for i in range(n_examples, len(axes_flat)):
                axes_flat[i].axis("off")

            # Add a common title
            fig.suptitle(
                f"{cluster_type} {cluster_id}: Examples (n={n_episodes})",
                fontsize=16,
            )

            # Set common labels for the figure
            fig.text(
                0.5,
                0.04,
                "Normalized Time (% of sleep episode)",
                ha="center",
                va="center",
            )
            fig.text(
                0.06,
                0.5,
                "Heart Rate (BPM)",
                ha="center",
                va="center",
                rotation="vertical",
            )

            # Use tight_layout with proper tuple for rect parameter
            fig.tight_layout(rect=(0.08, 0.08, 0.98, 0.95))
            fig.savefig(
                clusters_dir / f"{cluster_col}_{cluster_id}_hr_curves.png", dpi=300
            )
            plt.close(fig)

            # If nadir timing is available, create a separate nadir timing histogram
            if "nadir_time_pct" in cluster_df.columns:
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                # Create a DataFrame for the histplot to avoid type issues
                nadir_data = pd.DataFrame(
                    {"nadir_time_pct": cluster_df["nadir_time_pct"].values}
                )
                sns.histplot(
                    data=nadir_data, x="nadir_time_pct", bins=20, kde=True, ax=ax2
                )
                ax2.set_title(f"{cluster_type} {cluster_id}: Nadir Timing Distribution")
                ax2.set_xlabel("Nadir Time (% of sleep episode)")
                ax2.set_ylabel("Count")
                # Add vertical line for mean
                mean_nadir = cluster_df["nadir_time_pct"].mean()
                ax2.axvline(
                    x=mean_nadir,
                    color="red",
                    linestyle="--",
                    label=f"Mean: {mean_nadir:.2f}",
                )
                # Set x-axis limits from 0 to 1
                ax2.set_xlim(0, 1)
                ax2.legend()
                sns.despine()
                fig2.tight_layout()
                fig2.savefig(
                    clusters_dir / f"{cluster_col}_{cluster_id}_nadir_dist.png", dpi=300
                )
                plt.close(fig2)

    # Return summary stats as DataFrame
    return pd.DataFrame(summary_stats)


def perform_hr_statistics(hr_data_by_cluster: Dict[int, np.ndarray]) -> pd.DataFrame:
    """
    Perform statistical tests on heart rate data between clusters.

    Parameters
    ----------
    hr_data_by_cluster : Dict[int, np.ndarray]
        Dictionary mapping cluster IDs to arrays of HR curves

    Returns
    -------
    pd.DataFrame
        Summary statistics and test results
    """
    # Get all cluster IDs
    cluster_ids = list(hr_data_by_cluster.keys())

    # Prepare result storage
    results = []

    # Calculate basic statistics for each cluster
    for cluster_id, hr_curves in hr_data_by_cluster.items():
        mean_hr = np.mean(hr_curves, axis=0).mean()
        min_hr = np.mean([curve.min() for curve in hr_curves])
        max_hr = np.mean([curve.max() for curve in hr_curves])
        range_hr = max_hr - min_hr
        std_hr = np.mean(np.std(hr_curves, axis=1))

        results.append(
            {
                "cluster": cluster_id,
                "n_episodes": len(hr_curves),
                "mean_hr": mean_hr,
                "min_hr": min_hr,
                "max_hr": max_hr,
                "hr_range": range_hr,
                "std_hr": std_hr,
            }
        )

    # Create results DataFrame
    stats_df = pd.DataFrame(results)

    # Perform ANOVA to test for differences between clusters
    if len(cluster_ids) > 1:
        try:
            # Prepare data for ANOVA
            anova_data = []
            anova_groups = []

            for cluster_id, hr_curves in hr_data_by_cluster.items():
                cluster_means = np.mean(hr_curves, axis=1)
                anova_data.extend(cluster_means)
                anova_groups.extend([cluster_id] * len(cluster_means))

            # Perform one-way ANOVA, ensuring we use the groups correctly
            cluster_data = [
                np.mean(hr_data_by_cluster[cid], axis=1) for cid in cluster_ids
            ]
            anova_result = stats.f_oneway(*cluster_data)

            # Handle case where result might be a single value or array
            if hasattr(anova_result, "statistic"):
                f_stat = anova_result.statistic
                p_val = anova_result.pvalue
            else:
                f_stat = anova_result[0]
                p_val = anova_result[1]

            print(f"\nANOVA result for HR differences between clusters:")
            print(f"F-statistic: {float(f_stat):.3f}, p-value: {float(p_val):.5f}")

            if p_val < 0.05:
                print(
                    "There are significant differences in heart rates between clusters."
                )

                # Perform post-hoc tests (Tukey's HSD) if we have scipy stats version with this
                try:
                    from statsmodels.stats.multicomp import pairwise_tukeyhsd

                    posthoc = pairwise_tukeyhsd(anova_data, anova_groups, alpha=0.05)
                    print("\nPost-hoc Tukey HSD Test Results:")
                    print(posthoc)
                except ImportError:
                    print("statsmodels not available for Tukey HSD test")
            else:
                print("No significant differences in heart rates between clusters.")
        except Exception as e:
            print(f"Error performing ANOVA: {e}")
            print("Statistics calculation will continue without ANOVA results.")

    return stats_df


def compare_cluster_characteristics_inferential(
    df: pd.DataFrame,
    out_dir: Optional[Union[str, Path]],
    use_meta: bool = False,
    continuous_vars: Optional[List[str]] = None,
    categorical_vars: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Perform inferential statistical tests comparing clusters on key characteristics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with cluster assignments and characteristics
    out_dir : str or Path, optional
        Directory to save the results
    use_meta : bool, default=False
        Whether to use meta-clusters
    continuous_vars : List[str], optional
        List of continuous variables to test (defaults to common sleep parameters)
    categorical_vars : List[str], optional
        List of categorical variables to test (defaults to sex)

    Returns
    -------
    pd.DataFrame
        DataFrame with statistical test results
    """
    # Set default variables if not provided
    if continuous_vars is None:
        continuous_vars = [
            "age",
            "patient_chronotype",
            "se",
            "waso",
            "sfi",
            "sri",
            "nadir_time_pct",
            "nadir_hr",
            "episode_hr_mean",
            "chronotype_desync",
        ]
        # Filter to only included variables that exist
        continuous_vars = [var for var in continuous_vars if var in df.columns]

    if categorical_vars is None:
        categorical_vars = ["sex"]
        # Filter to only included variables that exist
        categorical_vars = [var for var in categorical_vars if var in df.columns]

    # Determine which cluster variable to use
    cluster_col = (
        "meta_cluster" if use_meta and "meta_cluster" in df.columns else "cluster"
    )
    cluster_type_title = "Meta-Cluster" if use_meta else "Cluster"

    # Create output directory if provided
    f_out = None
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)

        # Create a file to save results
        results_file = out_dir / f"{'meta_' if use_meta else ''}cluster_statistics.txt"
        f_out = open(results_file, "w")
        print(f"Statistical comparison of clusters ({cluster_col})", file=f_out)
        print("=" * 50, file=f_out)

    # Initialize results storage
    results = []

    # First, test if clusters differ on continuous variables
    for var in continuous_vars:
        if var not in df.columns:
            continue

        # Create a clean subset without NaN values
        subset = df[[cluster_col, var]].dropna()
        if len(subset) < 10:  # Skip if too few data points
            continue

        try:
            # Prepare data for ANOVA/Kruskal
            groups = [
                subset[subset[cluster_col] == c][var].values
                for c in sorted(subset[cluster_col].unique())
                if c != -1
            ]

            # Skip if any group is empty
            if any(len(g) == 0 for g in groups):
                continue

            # Convert any tuples to arrays and ensure numeric types
            groups_clean = [np.array(g, dtype=float) for g in groups]

            # Check if we have at least two groups for ANOVA
            if len(groups_clean) < 2:
                if f_out:
                    print(
                        f"\nSkipping {var}: Not enough clusters for statistical comparison (need at least 2, got {len(groups_clean)})",
                        file=f_out,
                    )
                continue

            # Try parametric test first
            f_stat, p_value = stats.f_oneway(*groups_clean)
            test_name = "ANOVA"
            formatted_p_value = _format_p_value_for_title(p_value)

            # If ANOVA assumptions violated, use non-parametric test
            if p_value < 0.05:
                # Also perform Kruskal-Wallis for non-parametric confirmation
                try:
                    # Ensure numeric conversion
                    kw_stat, kw_p = stats.kruskal(*groups_clean)
                    if kw_p < 0.05:
                        # Both tests significant, do post-hoc
                        from scikit_posthocs import posthoc_dunn

                        # Create DataFrame for posthoc test
                        data_for_posthoc = pd.DataFrame(
                            {
                                "value": np.concatenate(groups_clean),
                                "group": np.concatenate(
                                    [
                                        np.full(len(g), i)
                                        for i, g in enumerate(groups_clean)
                                    ]
                                ),
                            }
                        )
                        # Perform Dunn's test
                        posthoc_result = posthoc_dunn(
                            data_for_posthoc, val_col="value", group_col="group"
                        )

                        if f_out:
                            print(f"\nPost-hoc Dunn test for {var}:", file=f_out)
                            print(posthoc_result, file=f_out)
                except Exception as e:
                    # If posthoc test fails, just note the significant difference
                    if f_out:
                        print(f"Post-hoc test failed: {e}", file=f_out)

            # Store results
            result = {
                "variable": var,
                "test": test_name,
                "statistic": float(f_stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
            }
            results.append(result)

            if f_out:
                print(f"\n{test_name} for {var}:", file=f_out)
                print(
                    f"F-statistic: {float(f_stat):.3f}, p-value: {float(p_value):.5f}",
                    file=f_out,
                )
                if p_value < 0.05:
                    print(
                        f"Significant difference in {var} between clusters", file=f_out
                    )

                    # Add means by cluster
                    means = (
                        subset.groupby(cluster_col)[var]
                        .agg(["mean", "std"])
                        .reset_index()
                    )
                    print("\nMeans by cluster:", file=f_out)
                    print(means, file=f_out)

            # Generate and save boxplot if out_dir is provided
            if out_dir is not None and var in continuous_vars:
                # Filter out noise cluster (-1) for plotting
                plot_df = subset[
                    subset[cluster_col] != -1
                ].copy()  # Use .copy() to avoid SettingWithCopyWarning
                if not plot_df.empty:
                    plt.figure(figsize=(10, 6))
                    # Ensure cluster_col is treated as categorical for correct ordering and palette usage
                    # Convert to category if not already, especially if it's numeric
                    if not pd.api.types.is_categorical_dtype(plot_df[cluster_col]):
                        plot_df[cluster_col] = pd.Categorical(plot_df[cluster_col])

                    ax = sns.boxplot(
                        x=cluster_col, y=var, data=plot_df, palette="viridis"
                    )

                    # Calculate means for annotation
                    means = plot_df.groupby(cluster_col)[var].mean()

                    # Add mean annotations
                    for i, cat in enumerate(plot_df[cluster_col].cat.categories):
                        mean_val = means[
                            cat
                        ].item()  # Use .item() to get scalar from single-element Series/array
                        ax.text(
                            i,
                            mean_val,
                            f"{mean_val:.1f}",
                            horizontalalignment="center",
                            size="large",
                            color="white",
                            weight="semibold",
                            bbox=dict(
                                facecolor="black", alpha=0.5, boxstyle="round,pad=0.3"
                            ),
                        )

                    plt.title(
                        f'{var.replace("_", " ").title()} by {cluster_type_title} ({formatted_p_value})'
                    )
                    plt.xlabel(cluster_type_title)
                    plt.ylabel(var.replace("_", " ").title())
                    sns.despine()
                    plot_filename = out_dir / f"{var}_by_{cluster_col}_boxplot.png"
                    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
                    plt.close()
                    if f_out:
                        print(f"Saved boxplot to {plot_filename}", file=f_out)

        except Exception as e:
            if f_out:
                print(f"Error testing {var}: {str(e)}", file=f_out)

    # Test categorical variables using chi-square
    for var in categorical_vars:
        if var not in df.columns:
            continue

        try:
            # Create contingency table
            contingency = pd.crosstab(df[cluster_col], df[var])

            # Skip if table is too small
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                continue

            # Perform chi-square test
            chi2, p, dof, expected = stats.chi2_contingency(contingency)

            # Store results
            result = {
                "variable": var,
                "test": "Chi-square",
                "statistic": float(chi2),
                "p_value": float(p),
                "significant": p < 0.05,
            }
            results.append(result)

            if f_out:
                print(f"\nChi-square test for {var}:", file=f_out)
                print(
                    f"Chi2 = {float(chi2):.3f}, p-value: {float(p):.5f}, dof: {dof}",
                    file=f_out,
                )
                if p < 0.05:
                    print(
                        f"Significant association between {var} and cluster", file=f_out
                    )

                    # Add proportions
                    print("\nContingency table:", file=f_out)
                    print(contingency, file=f_out)
                    print("\nProportions (%):", file=f_out)
                    props = contingency.div(contingency.sum(axis=1), axis=0) * 100
                    print(props, file=f_out)

        except Exception as e:
            if f_out:
                print(f"Error testing {var}: {str(e)}", file=f_out)

    # Close output file if it exists
    if f_out:
        f_out.close()

    return pd.DataFrame(results)


def plot_fudolig_comparison(
    df: pd.DataFrame, out_path: Optional[Union[str, Path]] = None
):
    """
    Compare clustering results with Fudolig et al. approach by visualizing nadir timing distributions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with cluster assignments and nadir timing data
    out_path : str or Path, optional
        Path to save the plot
    """
    # Check if we have both clustering results and nadir timing
    if "k2_label" not in df.columns or "nadir_time_pct" not in df.columns:
        print("Cannot create Fudolig comparison: missing required columns")
        return

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Nadir timing distribution by k=2 cluster (Fudolig approach)
    sns.histplot(
        data=df,
        x="nadir_time_pct",
        hue="k2_label",
        kde=True,
        palette=["skyblue", "salmon"],
        ax=axes[0],
        bins=20,
        legend=True,
    )
    axes[0].set_title("Nadir Timing by K=2 Cluster\n(Fudolig et al. approach)")
    axes[0].set_xlabel("Nadir Time (% of sleep episode)")
    axes[0].set_ylabel("Count")

    # Get cluster means for legend
    k2_means = df.groupby("k2_label")["nadir_time_pct"].mean()
    axes[0].legend(
        title="K=2 Cluster",
        labels=[
            f"Cluster {i} (mean = {k2_means[i]:.2f})" for i in sorted(k2_means.index)
        ],
    )

    # Add vertical lines for means
    for i, mean_val in k2_means.items():
        color = "skyblue" if i == 0 else "salmon"
        axes[0].axvline(x=mean_val, color=color, linestyle="--", alpha=0.7)

    # Plot 2: Nadir timing distribution by multi-cluster solution
    if "cluster" in df.columns and len(df["cluster"].unique()) > 1:
        # Only include top 5 clusters for clarity
        cluster_counts = df["cluster"].value_counts()
        top_clusters = cluster_counts.index[:5]

        # Create a subset with only top clusters
        df_subset = df[df["cluster"].isin(top_clusters)]

        sns.histplot(
            data=df_subset,
            x="nadir_time_pct",
            hue="cluster",
            kde=True,
            palette="tab10",
            ax=axes[1],
            bins=20,
            legend=True,
        )
        axes[1].set_title("Nadir Timing by Multi-Cluster Solution\n(Current approach)")
        axes[1].set_xlabel("Nadir Time (% of sleep episode)")
        axes[1].set_ylabel("Count")

        # Get cluster means
        cluster_means = df_subset.groupby("cluster")["nadir_time_pct"].mean()

        # Custom legend
        axes[1].legend(
            title="Cluster",
            labels=[
                f"Cluster {i} (mean = {cluster_means[i]:.2f})"
                for i in sorted(cluster_means.index)
            ],
        )

        # Add vertical lines for means
        colors = sns.color_palette("tab10")  # Use seaborn's color_palette
        for i, (cluster, mean_val) in enumerate(cluster_means.items()):
            axes[1].axvline(
                x=mean_val, color=colors[i % len(colors)], linestyle="--", alpha=0.7
            )

    plt.tight_layout()

    # Save if requested
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")

    return fig, axes


def visualize_consistency_hr_profiles(
    eps_df: pd.DataFrame,
    patient_consistency: pd.DataFrame,
    low_consistency_ids: List[str],
    high_consistency_ids: List[str],
    n_plot: int,
    out_dir: Path,
):
    """
    Visualize HR profiles (hr_eq) for patients with low and high consistency.

    Parameters
    ----------
    eps_df : pd.DataFrame
        Episode dataframe containing 'id', 'cluster', and 'hr_eq'.
    patient_consistency : pd.DataFrame
        Patient consistency dataframe containing 'id' and 'cluster_consistency'.
    low_consistency_ids : List[str]
        List of patient IDs with the lowest consistency scores.
    high_consistency_ids : List[str]
        List of patient IDs with the highest consistency scores.
    n_plot : int
        Number of patients to plot for each group (low/high).
    out_dir : Path
        Directory to save the plot.
    """
    print(
        f"Generating HR profile plots for {n_plot} low and {n_plot} high consistency patients..."
    )

    # Ensure hr_eq is in a plottable format (list of numpy arrays)
    if not isinstance(eps_df["hr_eq"].iloc[0], np.ndarray):
        print(
            "Warning: 'hr_eq' column might not contain numpy arrays. Plotting may fail."
        )
        # Attempt conversion if possible, otherwise raise error or return
        try:
            # Example: Assuming hr_eq contains string representations of lists
            if isinstance(eps_df["hr_eq"].iloc[0], str):
                # Avoid DeprecationWarning: eval is deprecated, use ast.literal_eval
                import ast

                eps_df["hr_eq"] = eps_df["hr_eq"].apply(
                    lambda x: np.array(ast.literal_eval(x))
                )

        except Exception as e:
            print(
                f"Error converting hr_eq to plottable format: {e}. Skipping profile plot."
            )
            return

    # Determine unique clusters and create a color map
    all_clusters = sorted(eps_df["cluster"].unique())
    # Use a perceptually uniform colormap like 'viridis' or 'tab20' for distinct colors
    # Add gray for noise cluster (-1) if present
    colors = {}
    cmap = cm.get_cmap(
        "tab20", len(all_clusters)
    )  # Use tab20 for up to 20 distinct colors
    color_idx = 0
    for cluster_id in all_clusters:
        if cluster_id == -1:
            colors[cluster_id] = "grey"
        else:
            # Cycle through colormap, handle more clusters than colors if needed
            colors[cluster_id] = cmap(color_idx % cmap.N)
            color_idx += 1

    n_rows = 4
    n_cols = (
        n_plot * 2 + n_rows - 1
    ) // n_rows  # Calculate columns needed for 2*n_plot total plots

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3), sharex=True, sharey=True
    )
    axes = axes.flatten()  # Flatten to easily iterate

    plot_idx = 0
    patient_ids_to_plot = low_consistency_ids[:n_plot] + high_consistency_ids[:n_plot]
    group_labels = ["Low Consistency"] * n_plot + ["High Consistency"] * n_plot

    for patient_id, group_label in zip(patient_ids_to_plot, group_labels):
        if plot_idx >= len(axes):
            print(
                "Warning: More patients than available subplots. Some patients will not be plotted."
            )
            break

        ax = axes[plot_idx]
        patient_eps = eps_df[eps_df["id"] == patient_id]
        consistency_score = patient_consistency.loc[
            patient_consistency["id"] == patient_id, "cluster_consistency"
        ].iloc[0]

        if patient_eps.empty:
            ax.set_title(f"ID: {patient_id}\nNo Episodes Found")
            ax.text(
                0.5,
                0.5,
                "No Data",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            plot_idx += 1
            continue

        # Plot HR profiles for the patient
        legend_handles = {}  # To avoid duplicate legend entries per subplot
        for _, episode in patient_eps.iterrows():
            hr_profile = episode["hr_eq"]
            cluster_id = episode["cluster"]
            color = colors.get(
                cluster_id, "black"
            )  # Default to black if cluster somehow missing
            label = f"Cluster {cluster_id}"

            if isinstance(hr_profile, np.ndarray):
                # Check if time axis is needed or implicit
                time_axis = np.arange(
                    len(hr_profile)
                )  # Assume simple index if no time info
                (line,) = ax.plot(
                    time_axis,
                    hr_profile,
                    color=color,
                    alpha=0.7,
                    label=label if label not in legend_handles else "",
                )
                if label not in legend_handles:
                    legend_handles[label] = line
            else:
                print(
                    f"Skipping unplottable hr_profile for patient {patient_id}, episode index {_}"
                )

        ax.set_title(f"{group_label}\nID: {patient_id} (Cons: {consistency_score:.2f})")
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

        # Create legend only if there are handles
        if legend_handles:
            ax.legend(handles=legend_handles.values(), fontsize="small")

        plot_idx += 1

    # Hide unused axes
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        "HR Profiles for Low vs High Consistency Patients (Colored by Cluster)",
        fontsize=16,
    )
    fig.supxlabel("Time (Normalized/Resampled)", fontsize=12)
    fig.supylabel("Heart Rate (Equalized)", fontsize=12)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))  # Adjust layout to prevent title overlap

    plot_path = (
        out_dir / "consistency_analysis" / "hr_profiles_low_vs_high_consistency.png"
    )
    print(f"Saving HR profile comparison plot to: {plot_path}")
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
