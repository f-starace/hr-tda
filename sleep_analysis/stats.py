"""
Statistical functions for sleep data analysis.

This module provides statistical tests and analyses for
heart rate and sleep patterns.
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Typing imports for clarity if needed later for other functions
from typing import List, Dict, Optional, Tuple


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
        # Skip patients with only one episode
        if len(patient_df) < 2:
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
        # Ensure log2 argument is > 0 by checking if cluster_probs has any non-zero elements
        # And that min(...) is also > 0
        valid_probs_for_log = cluster_probs[cluster_probs > 0]
        if not valid_probs_for_log.empty:
            entropy = -np.sum(valid_probs_for_log * np.log2(valid_probs_for_log))
        else:
            entropy = 0.0

        min_val_for_max_entropy = min(len(cluster_counts), len(patient_df))
        max_entropy = (
            np.log2(min_val_for_max_entropy) if min_val_for_max_entropy > 0 else 0
        )

        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

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
            "chronotype_desync",
            "abs_chronotype_desync",
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

    if not patient_results:
        print(
            "Warning: No patients with multiple episodes found for consistency analysis"
        )
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

    print(f"\nSpearman correlations with cluster consistency (1-normalized entropy):")
    correlation_cols = [c for c in patient_df.columns if c.startswith("mean_")] + [
        "age",
        "patient_chronotype",
    ]
    correlation_cols = [c for c in correlation_cols if c in patient_df.columns]

    if correlation_cols:
        correlations = []
        for col in correlation_cols:
            if patient_df[col].nunique() > 1 and not patient_df[col].isna().all():
                # Ensure 'cluster_consistency' also has variance and no all NaNs
                if (
                    patient_df["cluster_consistency"].nunique() > 1
                    and not patient_df["cluster_consistency"].isna().all()
                ):

                    # temp_df for spearmanr to handle NaNs correctly within pairs
                    temp_df = patient_df[["cluster_consistency", col]].dropna()
                    if (
                        len(temp_df) >= 2
                    ):  # Need at least 2 pairs to calculate correlation
                        # Use try-except for robustness as spearmanr might fail
                        try:
                            corr_result = stats.spearmanr(
                                temp_df["cluster_consistency"], temp_df[col]
                            )
                            # Handle both tuple return and SpearmanrResult object return
                            if hasattr(corr_result, "correlation") and hasattr(
                                corr_result, "pvalue"
                            ):
                                # Newer scipy versions return an object
                                corr = float(corr_result.correlation)
                                p_val = float(corr_result.pvalue)
                                if pd.isna(corr) or pd.isna(p_val):
                                    print(
                                        f"Warning: spearmanr returned NaN for {col} (object access). Skipping."
                                    )
                                    continue
                            elif (
                                isinstance(corr_result, tuple) and len(corr_result) >= 2
                            ):
                                # Older versions return a tuple
                                try:
                                    corr = float(corr_result[0])
                                    p_val = float(corr_result[1])
                                    if pd.isna(corr) or pd.isna(p_val):
                                        print(
                                            f"Warning: spearmanr returned NaN for {col} (tuple access). Skipping."
                                        )
                                        continue
                                except (TypeError, ValueError):
                                    print(
                                        f"Could not convert spearmanr tuple results to float for {col}. Skipping."
                                    )
                                    continue
                            else:
                                print(
                                    f"Warning: Unexpected return type from spearmanr for {col} ({type(corr_result)}). Skipping."
                                )
                                continue  # Skip this variable

                            correlations.append(
                                {
                                    "variable": col,
                                    "correlation": corr,
                                    "p_value": p_val,
                                    "significant": p_val < 0.05,
                                }
                            )
                        except Exception as e_corr:
                            print(
                                f"Error calculating Spearman correlation for {col}: {e_corr}"
                            )
                    else:
                        print(
                            f"Skipping correlation for {col} due to insufficient non-NaN pairs."
                        )
                else:
                    print(
                        f"Skipping correlation for {col} because 'cluster_consistency' lacks variance or is all NaN."
                    )

            else:
                print(
                    f"Skipping correlation for {col} due to lack of variance or all NaN values."
                )

        if correlations:
            corr_df = pd.DataFrame(correlations)
            print(corr_df.sort_values("p_value").to_string(index=False))

    try:
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
                clean_pred = pred.replace("-", "_").replace(
                    "%", "pct"
                )  # Patsy friendly names
                if (
                    patient_df[pred].dtype == "object" or patient_df[pred].nunique() < 5
                ):  # Heuristic for categorical
                    formula_parts.append(f"C({clean_pred})")
                else:
                    formula_parts.append(clean_pred)

            # Rename columns in a copy of the dataframe for patsy
            patsy_df = patient_df.copy()
            patsy_df.columns = [
                col.replace("-", "_").replace("%", "pct") for col in patsy_df.columns
            ]

            formula = "cluster_consistency ~ " + " + ".join(formula_parts)
            model = smf.ols(
                formula,
                data=patsy_df.dropna(subset=predictors + ["cluster_consistency"]),
            ).fit()  # dropna for OLS
            print("\nOLS Model predicting cluster consistency:")
            print(
                f"R-squared: {model.rsquared:.3f}, Adj R-squared: {model.rsquared_adj:.3f}"
            )
            print("Significant predictors (p<0.05):")
            for var, p_value in model.pvalues.items():
                if p_value < 0.05 and var != "Intercept":
                    coef = model.params[var]
                    print(f"  {var}: coef={coef:.4f}, p={p_value:.4f}")
    except Exception as e:
        print(f"Could not fit OLS model: {e}")

    return patient_df
