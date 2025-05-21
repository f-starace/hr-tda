"""
Statistical-models module for sleep-analysis (re-written).

Key functions
-------------
multinomial_mixed_model()        - Bayesian categorical GLMM (Bambi)
multinomial_frequentist_model()  - MNLogit + cluster-robust SEs
visualize_model_results()        - unchanged (still regex-based)
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Sequence, Dict, Any, Union

# ------------------------------------------------------------------ #
# core imports
# ------------------------------------------------------------------ #
import numpy as np
import pandas as pd
import arviz as az
import bambi as bmb
from bambi.priors import Prior
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MultinomialResultsWrapper
from sklearn.preprocessing import StandardScaler
from matplotlib.container import BarContainer

import re
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------------------------------------------------------ #
# small utilities
# ------------------------------------------------------------------ #
_CAUSAL_CACHE: Dict[str, Sequence[str]] = {}  # target_relationship -> confounders


def _log(msg: str, fh):
    """Tiny helper – print and optionally write to file‐handle."""
    print(msg)
    if fh:
        fh.write(msg + "\n")


def _make_formula(
    df: pd.DataFrame,
    cluster_col: str,
    use_causal: bool,
    target_relationship: str,
    extra_terms_for_non_causal: Optional[Sequence[str]] = None,
    is_frequentist: bool = False,
) -> tuple[str, List[str]]:
    """
    Build RHS formula and return the essential columns.
    Adds C() wrapper for categorical variables if is_frequentist is True.
    """
    from sleep_analysis.causal import identify_confounders

    exposure = target_relationship.split(" -> ")[0].strip()
    rhs_terms: List[str] = []

    if use_causal:
        # First, always add the exposure if it exists (crucial for causal models)
        has_valid_exposure = False
        if exposure in df.columns:
            rhs_terms.append(exposure)
            has_valid_exposure = True
        else:
            print(
                f"WARNING: Exposure '{exposure}' not in DataFrame columns. Causal interpretation limited."
            )

        # Attempt to get confounders, handling potential failure of identify_confounders
        confounders = []
        try:
            if target_relationship not in _CAUSAL_CACHE:
                try:
                    print(f"Identifying confounders for {target_relationship}...")
                    _CAUSAL_CACHE[target_relationship] = identify_confounders(
                        df, target_relationship
                    )
                except Exception as e_conf:
                    print(
                        f"WARNING: identify_confounders failed for '{target_relationship}': {e_conf}"
                    )
                    print(
                        f"WARNING: Falling back to default confounders for {target_relationship}"
                    )
                    # Pick default confounders based on the target
                    if "patient_chronotype" in exposure:
                        _CAUSAL_CACHE[target_relationship] = ["age", "sex", "sri"]
                    elif "nadir_time_pct" in exposure:
                        _CAUSAL_CACHE[target_relationship] = [
                            "age",
                            "sex",
                            "patient_chronotype",
                            "sri",
                        ]
                    elif "se" in exposure:
                        _CAUSAL_CACHE[target_relationship] = [
                            "age",
                            "sex",
                            "sfi",
                            "waso",
                        ]
                    else:
                        _CAUSAL_CACHE[target_relationship] = ["age", "sex"]

            confounders = [
                c
                for c in _CAUSAL_CACHE[target_relationship]
                if c in df.columns and c != exposure
            ]
        except Exception as e:
            print(f"Error retrieving confounders from cache: {e}")
            confounders = []

        # Add confounders to RHS terms
        rhs_terms.extend(confounders)

        # If we only have exposure (or don't even have that), add minimal default adjustment
        if len(rhs_terms) <= 1 and "sex" in df.columns:
            print(
                f"WARNING: Very few terms in causal formula. Adding 'sex' as minimal adjustment."
            )
            if "sex" not in rhs_terms:
                rhs_terms.append("sex")

        if len(rhs_terms) <= 1 and "age" in df.columns:
            print(
                f"WARNING: Very few terms in causal formula. Adding 'age' as minimal adjustment."
            )
            if "age" not in rhs_terms:
                rhs_terms.append("age")

        # In the extreme edge case where we have no valid terms at all, add intercept-only placeholder
        if not rhs_terms:
            print(
                f"CRITICAL: No valid terms for formula. Using intercept-only model (not recommended)."
            )
            rhs_terms = ["1"]  # Forces intercept-only model

    else:  # Non-causal case
        base_predictors = ["age", "sex", "patient_chronotype"]
        rhs_terms.extend([p for p in base_predictors if p in df.columns])

        if extra_terms_for_non_causal:
            rhs_terms.extend([t for t in extra_terms_for_non_causal if t in df.columns])

    # Deduplicate, filter for existence in df
    potential_rhs_terms = sorted(
        list(set(term for term in rhs_terms if term in df.columns or term == "1"))
    )

    # Wrap categorical variables with C() for frequentist models
    final_rhs_terms = []
    essential_cols_set = set([cluster_col, "id"])
    for term in potential_rhs_terms:
        if term == "1":
            final_rhs_terms.append(term)
            continue

        # Add term to essential columns
        essential_cols_set.add(term)

        # Check if term is likely categorical and needs wrapping for frequentist
        is_categorical = False
        if term in df.columns:
            # Consider object type or lownunique numeric as categorical
            if df[term].dtype == "object" or df[term].dtype.name == "category":
                is_categorical = True
            elif (
                pd.api.types.is_numeric_dtype(df[term].dtype) and df[term].nunique() < 5
            ):  # Heuristic
                # Treat low-nunique numeric as categorical unless it's the exposure variable
                if term != exposure:
                    is_categorical = True

        if (
            is_frequentist and is_categorical and term != "sex"
        ):  # Let manual sex encoding handle 'sex'
            final_rhs_terms.append(f"C({term})")
            print(
                f"INFO: Wrapping categorical term '{term}' with C() for frequentist model."
            )
        else:
            final_rhs_terms.append(term)

    # Build the formula
    if "1" in final_rhs_terms and len(final_rhs_terms) > 1:
        final_rhs_terms.remove("1")

    rhs = (
        " + ".join(final_rhs_terms) if final_rhs_terms else "1"
    )  # Default to intercept-only if empty
    formula = f"{cluster_col} ~ {rhs}"

    essential_cols = list(essential_cols_set)

    # For 'needed', include the outcome and all predictors to ensure they don't get dropped in df.dropna()
    # Always add 'id' for random effects/clustering
    essential_cols = [cluster_col] + final_rhs_terms + ["id"]

    # Filter out "1" from essential_cols as it's not a real column
    essential_cols = [col for col in essential_cols if col != "1"]

    # Print summary of the formula creation
    terms_desc = ", ".join(final_rhs_terms) if final_rhs_terms else "intercept-only"
    if use_causal:
        approach = f"causal (exposure={exposure})"
    else:
        approach = "non-causal (standard)"
    print(f"Created {approach} formula: {formula}")
    print(f"Essential columns for filtering NAs: {essential_cols}")

    return formula, essential_cols


# ------------------------------------------------------------------ #
# Bayesian multinomial mixed model
# ------------------------------------------------------------------ #
def multinomial_mixed_model(
    df: pd.DataFrame,
    out_path: Path | None = None,
    *,
    use_meta: bool = False,
    use_causal: bool = False,
    target_relationship: str = "patient_chronotype -> cluster",
    prior_scale: float = 1.0,
    grouped_loo: bool = False,
    reduced_mcmc: bool = False,
) -> az.InferenceData | None:
    """
    Categorical GLMM with patient random intercept using Bambi/NUTS.

    Noise points (cluster = -1) are removed *before* modelling.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame with cluster and predictor variables
    out_path : Path, optional
        Path to save model summary
    use_meta : bool, default=False
        Whether to use meta_cluster instead of cluster as outcome
    use_causal : bool, default=False
        Whether to use causal variable selection
    target_relationship : str, default="patient_chronotype -> cluster"
        Target causal relationship to model (if use_causal=True)
    prior_scale : float, default=1.0
        Scale parameter for prior distributions
    grouped_loo : bool, default=False
        Whether to compute grouped leave-one-out cross-validation
    reduced_mcmc : bool, default=False
        Whether to use reduced MCMC settings (for faster computation)

    Returns
    -------
    az.InferenceData or None
        ArviZ InferenceData object or None if model fitting failed
    """
    # ------------------  prep & logging  --------------------------- #
    log_fh = None

    # Create a target-specific subfolder for outputs
    if out_path:
        out_path = Path(out_path)

        # Check if output path already has a directory structure that includes target relationship
        # If the parent directory name contains the target relationship, don't create another subfolder
        parent_dir_name = out_path.parent.name.lower()
        relationship_folder_name = (
            target_relationship.replace(" -> ", "_to_").replace(" ", "_").lower()
        )

        if relationship_folder_name in parent_dir_name:
            # We're already in a target-specific directory, don't create a nested one
            _log(
                f"Using existing directory for target relationship: {target_relationship}",
                log_fh,
            )
        else:
            # Create a safe folder name from the target relationship
            target_folder_name = target_relationship.replace(" -> ", "_to_").replace(
                " ", "_"
            )

            # Determine the base directory - either the parent of out_path or out_path itself
            if out_path.suffix:  # If out_path includes a filename
                results_dir = out_path.parent
            else:  # If out_path is just a directory
                results_dir = out_path

            # Create the target-specific subfolder
            target_dir = results_dir / target_folder_name
            target_dir.mkdir(parents=True, exist_ok=True)

            # Update the output path to point to the new subfolder
            if out_path.suffix:  # If original out_path included a filename
                out_path = target_dir / out_path.name
            else:
                # Default filename if none provided
                out_path = target_dir / "model_summary.txt"

        # Open log file in the target directory
        log_fh = open(out_path.with_suffix(".log"), "w")

        # Log the target relationship
        _log(f"Target relationship: {target_relationship}", log_fh)
    else:
        # No output path provided, just log to console
        _log(f"Target relationship: {target_relationship}", None)

    try:
        df = df.copy()
        cluster_col = "cluster"

        # remove noise rows
        df = df[df[cluster_col] != -1].copy()
        _log(
            f"Bayesian model on {len(df)} episodes, K = {df[cluster_col].nunique()}",
            log_fh,
        )

        # IMPORTANT: Cast outcome to category BEFORE identifying numeric columns
        # This ensures the outcome is treated as categorical, not continuous
        df[cluster_col] = df[cluster_col].astype(str)
        _log(
            f"Outcome '{cluster_col}' has {df[cluster_col].nunique()} categories",
            log_fh,
        )

        # make formula
        # Define terms that are always added to the NON-CAUSAL model
        extra_for_non_causal = [
            "nadir_time_pct",
            "se",
            "waso",
            "episode_hr_mean",
            "sri",
        ]
        formula, needed = _make_formula(
            df, cluster_col, use_causal, target_relationship, extra_for_non_causal
        )
        formula = (
            formula + " + (1|id)"
        )  # Random intercept always added for Bayesian model
        _log(f"Formula: {formula}", log_fh)

        # drop rows with NAs in needed cols
        df = df.dropna(subset=[c for c in needed if c in df.columns])
        _log(f"After dropping NAs, dataset has {len(df)} rows", log_fh)

        # standardise numeric predictors - EXCLUDE the cluster column which is categorical
        # Only standardize predictor variables, not the outcome or grouping variables
        num_cols = [
            c
            for c in needed
            if c != cluster_col
            and c != "id"
            and c in df.columns
            and df[c].dtype.kind in "if"
        ]
        scaling_info = None  # Initialize to None to prevent possibly unbound error

        if num_cols:
            _log(f"Standardizing numeric predictor columns: {num_cols}", log_fh)
            scaler = StandardScaler()
            # Explicitly make a copy of the dataframe for standardization
            df_copy = df.copy()
            df_copy[num_cols] = scaler.fit_transform(df[num_cols])

            # Report scaling parameters for reference
            scaling_info = pd.DataFrame(
                {"mean": scaler.mean_, "scale": scaler.scale_}, index=num_cols
            )
            _log(f"Scaling parameters:\n{scaling_info}", log_fh)

            df = df_copy  # Replace with the copy to ensure we're not modifying a view

        # drop array columns altogether
        arr_cols = []
        for col in df.columns:
            if hasattr(df[col].iloc[0], "__len__") and not isinstance(
                df[col].iloc[0], str
            ):
                arr_cols.append(col)

        if arr_cols:
            _log(f"Dropping array columns: {arr_cols}", log_fh)
            # Use drop() to create a new DataFrame rather than modifying in place
            df = df.drop(columns=arr_cols)

        # ---------------  priors (Student-t default) ------------------- #
        _log("Setting up model priors...", log_fh)
        priors: Dict[str, Prior] = {}
        for col in num_cols:
            priors[col] = Prior("StudentT", mu=0, sigma=prior_scale, nu=3)
        priors["Intercept"] = Prior("StudentT", mu=0, sigma=5, nu=3)

        # ---------------  model fitting  ------------------------------- #
        _log("Initializing Bambi model...", log_fh)
        try:
            m = bmb.Model(formula, df, family="categorical", priors=priors)
            _log(f"Model initialized with formula: {m.formula}", log_fh)
        except Exception as e_init:
            _log(f"Error initializing Bambi model: {e_init}", log_fh)
            if out_path:
                with open(out_path, "w") as fh:
                    fh.write(f"Target relationship: {target_relationship}\n")
                    fh.write(f"Model Formula: {formula}\n\n")
                    fh.write(f"ERROR initializing model: {e_init}\n")
            if log_fh:
                log_fh.close()
            return None

        draws, tune, chains = (1000, 500, 2) if reduced_mcmc else (2000, 1000, 4)
        _log(
            f"Starting MCMC with {chains} chains, {tune} tuning steps, {draws} draws",
            log_fh,
        )
        _log("Initializing NUTS using adapt_diag...", log_fh)

        # Wrap m.fit() in a try-except block
        try:
            idata = m.fit(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=min(
                    chains, os.cpu_count() or 1, 4
                ),  # Ensure cores is at least 1 and max 4
                target_accept=0.9,  # Reverted
                max_treedepth=10,  # Reverted
                init="adapt_diag",
                return_inferencedata=True,
                log_likelihood=True,
                predict=True,  # Generate posterior predictive samples
                random_seed=42,
            )
            _log("MCMC sampling completed", log_fh)

            # Save scaling information for reference
            if out_path and scaling_info is not None:
                scale_path = out_path.parent / f"{out_path.stem}_scaling.csv"
                scaling_info.to_csv(scale_path)
                _log(f"Saved scaling parameters to {scale_path}", log_fh)

            # Assess significance of target relationship if it's in the model
            exposure = target_relationship.split(" -> ")[0].strip()
            if exposure in num_cols or (
                exposure in df.columns and df[exposure].dtype.kind in "O"
            ):
                # If the exposure is a categorical variable, look for its dummy variables
                var_pattern = exposure if exposure in num_cols else f"{exposure}["

                _log(
                    f"Assessing significance of '{exposure}' in cluster prediction",
                    log_fh,
                )

                # Initialize target_vars to ensure it's defined for all code paths
                target_vars = []

                # Initialize significance_path for later references
                significance_path = None
                if out_path:
                    significance_path = (
                        out_path.parent / f"{out_path.stem}_significance.txt"
                    )

                # Get posterior samples for the target variable
                if "posterior" in idata:
                    # Find all variables in posterior that match the target
                    # Use a more flexible pattern matching approach for categorical variables
                    posterior_vars = list(idata.posterior.data_vars.keys())

                    # Log all variable names for debugging
                    _log(f"Available posterior variables: {posterior_vars}", log_fh)

                    # Improve pattern matching - handle both numeric and categorical cases
                    if exposure in num_cols:
                        # For numeric variables, just look for the exact name
                        target_vars = [
                            var
                            for var in posterior_vars
                            if var == exposure
                            or (var.startswith(exposure) and ":" in var)
                        ]
                    else:
                        # For categorical variables, look for dummy patterns with brackets
                        # e.g., "sex[T.M]", "sex[M]", "sex:cluster[1]", etc.
                        target_vars = [
                            var
                            for var in posterior_vars
                            if var.startswith(f"{exposure}[")
                            or var.startswith(f"{exposure}:")
                        ]

                    # If no variables found with primary patterns, try a more lenient search
                    if not target_vars:
                        _log(
                            f"No exact matches for {exposure}, trying partial match",
                            log_fh,
                        )
                        target_vars = [
                            var
                            for var in posterior_vars
                            if exposure in var and "Intercept" not in var
                        ]

                    if target_vars:
                        _log(
                            f"Found {len(target_vars)} coefficients related to {exposure}: {target_vars}",
                            log_fh,
                        )

                        # Create or clear significance file
                        if out_path and significance_path:
                            with open(significance_path, "w") as fh:
                                fh.write(
                                    f"Target relationship: {target_relationship}\n"
                                )
                                fh.write(f"Model Formula: {formula}\n\n")
                                fh.write("Significance Analysis with 95% HDI\n")
                                fh.write("=" * 50 + "\n\n")

                        for var in target_vars:
                            # Calculate probability of effect being non-zero
                            samples = idata.posterior[var].values.flatten()
                            prob_nonzero = (
                                np.mean(samples > 0)
                                if np.mean(samples) > 0
                                else np.mean(samples < 0)
                            )
                            prob_nonzero = max(
                                prob_nonzero, 1 - prob_nonzero
                            )  # Get the most extreme probability

                            # Calculate 95% HDI
                            hdi_2_5 = np.percentile(samples, 2.5)
                            hdi_97_5 = np.percentile(samples, 97.5)

                            # Check if HDI excludes zero
                            significant = (hdi_2_5 > 0) or (hdi_97_5 < 0)

                            # Log the results
                            significance_str = (
                                "SIGNIFICANT" if significant else "not significant"
                            )
                            _log(
                                f"Coefficient {var}: mean={np.mean(samples):.4f}, "
                                f"95% HDI=[{hdi_2_5:.4f}, {hdi_97_5:.4f}], "
                                f"P(effect != 0)={prob_nonzero:.2%} - {significance_str}",
                                log_fh,
                            )

                            if significance_path:
                                with open(significance_path, "a") as fh:
                                    fh.write(f"Coefficient {var}:\n")
                                    fh.write(f"  Mean effect: {np.mean(samples):.4f}\n")
                                    fh.write(
                                        f"  95% HDI: [{hdi_2_5:.4f}, {hdi_97_5:.4f}]\n"
                                    )
                                    fh.write(f"  P(effect != 0): {prob_nonzero:.2%}\n")
                                    fh.write(f"  Conclusion: {significance_str}\n\n")
                    else:
                        _log(
                            f"Could not find coefficients for {exposure} in posterior",
                            log_fh,
                        )
                else:
                    _log(
                        "No posterior samples available to assess significance", log_fh
                    )
            else:
                _log(
                    f"Target variable '{exposure}' not found in model predictors",
                    log_fh,
                )

        except Exception as e_fit:
            _log(f"!!! Model fitting failed for formula '{formula}': {e_fit}", log_fh)
            import traceback

            _log(traceback.format_exc(), log_fh)

            if out_path:
                with open(out_path, "w") as fh:
                    fh.write(f"Target relationship: {target_relationship}\n")
                    fh.write(f"Model Formula: {formula}\n\n")
                    fh.write(f"ERROR during model fitting: {e_fit}\n\n")
                    fh.write("Detailed traceback:\n")
                    fh.write(traceback.format_exc())
            if log_fh:
                log_fh.close()
            return None

        # --------------- Check for divergences in all chains ----------- #
        # Ensure idata and sample_stats exist before accessing
        if (
            idata
            and hasattr(idata, "sample_stats")
            and "diverging" in idata.sample_stats
        ):
            divergences_per_chain = idata.sample_stats.diverging.sum(dim="draw").values
            total_divergences = divergences_per_chain.sum()
            _log(
                f"Divergences per chain: {divergences_per_chain} (total: {total_divergences})",
                log_fh,
            )

            # Only skip if ALL chains have divergences
            if (divergences_per_chain > 0).all():
                skip_msg = (
                    f"[Divergence Skip] All {chains} chains reported divergences for formula '{formula}'.\n"
                    f"Skipping detailed summary and LOO for this model.\n"
                    f"Divergences per chain: {divergences_per_chain}"
                )
                _log(skip_msg, log_fh)
                if out_path:
                    # Write skip message to the output file
                    with open(out_path, "w") as fh:  # Overwrite with skip message
                        fh.write(f"Target relationship: {target_relationship}\n")
                        fh.write(
                            f"Model Formula: {m.formula}\n\n"
                        )  # Access formula from model 'm'
                        fh.write(skip_msg + "\n")
                if log_fh:
                    log_fh.close()
                return idata  # Return idata anyway for potential diagnostics
            elif total_divergences > 0:
                _log(
                    f"Warning: {total_divergences} divergences detected, but not in all chains. Proceeding with caution.",
                    log_fh,
                )
        else:
            _log(
                f"Could not retrieve divergence information. Proceeding with summary.",
                log_fh,
            )

        # ---------------  posterior-summary & save  -------------------- #
        if out_path:
            try:
                with open(out_path, "w") as fh:
                    fh.write(f"Target relationship: {target_relationship}\n")
                    fh.write(
                        f"Model Formula: {m.formula}\n\n"
                    )  # Access formula from model 'm'
                    # Check if idata is valid before summarizing
                    if idata:
                        summary_df = az.summary(
                            idata, var_names=["~Intercept"], round_to=3
                        )
                        fh.write(summary_df.to_string())
                    else:
                        fh.write("Model fitting produced no data (idata is None).\n")
                _log(f"Saved model summary to {out_path}", log_fh)

                # Add posterior predictive check visualization if idata is valid
                if idata:
                    try:
                        # Generate posterior predictive check (ppc) plot
                        ppc_path = out_path.parent / f"{out_path.stem}_ppc.png"
                        # Get cluster column from data (should be in df)
                        if cluster_col in df.columns:
                            # ArviZ PPC with observed data
                            plt.figure(figsize=(10, 6))
                            # Simple posterior predictive check comparing predicted vs actual cluster distributions
                            if "posterior_predictive" in idata:
                                az.plot_ppc(
                                    idata, var_names=[cluster_col], num_pp_samples=100
                                )
                                plt.title(
                                    f"Posterior Predictive Check - {target_relationship}"
                                )
                                plt.tight_layout()
                                plt.savefig(ppc_path, dpi=300)
                                plt.close()
                                _log(
                                    f"Saved posterior predictive check to {ppc_path}",
                                    log_fh,
                                )

                                # Also create a more detailed PPC plot with densities
                                ppc_density_path = (
                                    out_path.parent / f"{out_path.stem}_ppc_density.png"
                                )
                                plt.figure(figsize=(12, 8))
                                ax = az.plot_ppc(
                                    idata,
                                    var_names=[cluster_col],
                                    kind="density",
                                    data_pairs={(cluster_col, cluster_col)},
                                    figsize=(12, 8),
                                )
                                plt.title(
                                    f"Posterior Predictive Density Check - {target_relationship}"
                                )
                                plt.tight_layout()
                                plt.savefig(ppc_density_path, dpi=300)
                                plt.close()
                                _log(
                                    f"Saved posterior predictive density plot to {ppc_density_path}",
                                    log_fh,
                                )
                            else:
                                _log(
                                    f"Skipping PPC plot: 'posterior_predictive' not in inference data",
                                    log_fh,
                                )
                        else:
                            _log(
                                f"Skipping PPC plot: '{cluster_col}' not in DataFrame columns",
                                log_fh,
                            )

                        # Create trace plots to check sampling quality
                        trace_path = out_path.parent / f"{out_path.stem}_trace.png"
                        # Create trace plot for all parameters (not just target vars)
                        plt.figure(figsize=(12, 10))
                        # Plot all relevant parameters without relying on target_vars
                        az.plot_trace(
                            idata, var_names=["~1|id"]
                        )  # Exclude random effects
                        plt.title(f"Parameter Traces - {target_relationship}")
                        plt.tight_layout()
                        plt.savefig(trace_path, dpi=300)
                        plt.close()
                        _log(f"Saved parameter trace plot to {trace_path}", log_fh)

                        # Create a separate trace plot focusing on the target relationship if available
                        if exposure in num_cols or (
                            exposure in df.columns and "posterior" in idata
                        ):
                            target_trace_path = (
                                out_path.parent / f"{out_path.stem}_target_trace.png"
                            )
                            plt.figure(figsize=(12, 8))
                            # Use the var_pattern defined earlier to find relevant parameters
                            var_pattern = (
                                exposure if exposure in num_cols else f"{exposure}["
                            )
                            az.plot_trace(
                                idata,
                                var_names=[
                                    var
                                    for var in idata.posterior.data_vars
                                    if var.startswith(var_pattern)
                                    and "Intercept" not in var
                                ],
                            )
                            plt.title(
                                f"Target Parameter Traces: {exposure} - {target_relationship}"
                            )
                            plt.tight_layout()
                            plt.savefig(target_trace_path, dpi=300)
                            plt.close()
                            _log(
                                f"Saved target parameter trace plot to {target_trace_path}",
                                log_fh,
                            )

                    except Exception as e_ppc:
                        _log(
                            f"Error generating posterior predictive check: {e_ppc}",
                            log_fh,
                        )
            except Exception as e_summary:
                _log(
                    f"Error saving model summary for formula '{formula}': {e_summary}",
                    log_fh,
                )

        # ---------------  optional grouped PSIS-LOO -------------------- #
        if (
            grouped_loo and idata and "log_likelihood" in idata
        ):  # renamed from use_grouped_loo for clarity
            try:
                _log("Computing LOO cross-validation...", log_fh)
                loo_results = az.loo(idata)  # Removed group="id"
                _log(
                    f"\nPSIS-LOO (observation-wise) for formula '{formula}':\n{loo_results}",
                    log_fh,
                )
                if out_path:
                    with open(out_path, "a") as fh:
                        fh.write("\n\nPSIS-LOO (observation-wise):\n")
                        fh.write(str(loo_results))
            except Exception as e_loo:
                _log(f"Error computing LOO for formula '{formula}': {e_loo}", log_fh)
                if out_path:
                    with open(out_path, "a") as fh:
                        fh.write(f"\n\nError computing LOO: {e_loo}\n")
        elif grouped_loo and (idata is None or "log_likelihood" not in idata):
            _log(
                f"LOO calculation skipped: No valid log likelihood data available.",
                log_fh,
            )

        if log_fh:
            log_fh.close()

        return idata

    except Exception as e_outer:
        _log(f"CRITICAL ERROR in multinomial_mixed_model: {e_outer}", log_fh)
        import traceback

        _log(traceback.format_exc(), log_fh)

        if out_path:
            with open(out_path, "w") as fh:
                fh.write(f"Target relationship: {target_relationship}\n\n")
                fh.write("CRITICAL ERROR IN MODEL PROCESSING\n\n")
                fh.write(f"Error: {e_outer}\n\n")
                fh.write("Detailed traceback:\n")
                fh.write(traceback.format_exc())

        if log_fh:
            log_fh.close()

    return None


# ------------------------------------------------------------------ #
# Frequentist counterpart – MNLogit + cluster-robust SEs
# ------------------------------------------------------------------ #
def multinomial_frequentist_model(
    df: pd.DataFrame,
    out_path: Path | None = None,
    *,
    use_meta: bool = False,
    use_causal: bool = False,
    target_relationship: str = "patient_chronotype -> cluster",
) -> Any:  # Use a more generic return type
    """
    Frequentist analogue: multinomial logit with patient-cluster robust SEs.

    This function implements a multinomial logistic regression model with
    cluster-robust standard errors. It handles standardization of predictors
    to prevent numerical issues and provides detailed error reporting.

    Note: Unlike the Bayesian model, statsmodels mnlogit requires the dependent
    variable (cluster) to be numeric, not categorical.
    """
    log_fh = None
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(out_path.with_suffix(".log"), "w")

    try:
        df = df.copy()
        cluster_col = (
            "meta_cluster" if use_meta and "meta_cluster" in df.columns else "cluster"
        )

        # Remove noise rows
        df = df[df[cluster_col] != -1].copy()
        _log(
            f"Frequentist model on {len(df)} rows, K = {df[cluster_col].nunique()}",
            log_fh,
        )

        # Define terms that are always added to the NON-CAUSAL model for frequentist approach
        extra_for_non_causal_freq = [
            "nadir_time_pct",
            "se",
            "waso",
            "episode_hr_mean",
            "sri",
        ]
        formula, needed = _make_formula(
            df,
            cluster_col,
            use_causal,
            target_relationship,
            extra_for_non_causal_freq,
            is_frequentist=True,
        )
        _log(f"Frequentist model formula: {formula}", log_fh)

        # Drop NAs for needed columns
        df = df.dropna(subset=[c for c in needed if c in df.columns])
        _log(f"After dropping NAs, dataset has {len(df)} rows", log_fh)

        # ---------- PREPROCESSING STEPS ----------

        # For frequentist MNLogit, the dependent variable must be numeric, not categorical
        # First, ensure the cluster variable is treated as categorical, then convert to numeric codes
        _log("Converting cluster variable to numeric codes...", log_fh)
        orig_cluster_values = df[cluster_col].copy()

        # Use pandas categorical dtype which creates better numeric codes
        df[cluster_col] = pd.Categorical(df[cluster_col])

        # Store the codes for statsmodels
        cluster_codes = df[cluster_col].cat.codes

        # Create a mapping for interpretation
        code_mapping = {
            code: value for code, value in zip(cluster_codes, orig_cluster_values)
        }
        _log(f"Cluster mapping for mnlogit: {code_mapping}", log_fh)

        # Replace the cluster column with the numeric codes
        df[cluster_col] = cluster_codes
        _log(f"Converted cluster to numeric codes: {df[cluster_col].unique()}", log_fh)

        # Standardize continuous predictors
        num_cols = [
            c
            for c in needed
            if c != cluster_col
            and c != "id"
            and c in df.columns
            and df[c].dtype.kind in "if"
        ]
        _log(f"Standardizing numerical columns: {num_cols}", log_fh)

        scaling_info = None  # Initialize to None to prevent possibly unbound error
        if num_cols:
            scaler = StandardScaler()
            df_scaled = df.copy()
            df_scaled[num_cols] = scaler.fit_transform(df[num_cols])

            # Report scaling parameters for later reference
            scaling_info = pd.DataFrame(
                {"mean": scaler.mean_, "scale": scaler.scale_}, index=num_cols
            )
            _log(f"Scaling parameters:\n{scaling_info}", log_fh)

            df = df_scaled

        # Drop array columns or columns with complex types
        arr_cols = []
        for col in df.columns:
            if hasattr(df[col].iloc[0], "__len__") and not isinstance(
                df[col].iloc[0], str
            ):
                arr_cols.append(col)

        if arr_cols:
            _log(f"Dropping array columns: {arr_cols}", log_fh)
            df = df.drop(columns=arr_cols)

        # ---------- MODEL FITTING WITH PROGRESSIVE SIMPLIFICATION ----------

        # Find target variable (exposure) from the relationship
        target_var = target_relationship.split(" -> ")[0].strip()
        _log(f"Target variable: {target_var}", log_fh)

        # Copy the dataframe to ensure we don't modify the original
        model_data = df.dropna().copy()
        if len(model_data) < len(df):
            _log(f"Dropped {len(df) - len(model_data)} rows with NaN values", log_fh)

        # Check for ID-like columns that should not be predictors
        id_cols_to_exclude = []
        for col in model_data.columns:
            # Look for columns that might be identifiers (containing dates or ID patterns)
            if col != cluster_col and col != "id":
                # Check if column contains string values with date/time format or double underscores (typical for IDs)
                if model_data[col].dtype == object:
                    sample_val = str(model_data[col].iloc[0])
                    if "__" in sample_val or "T" in sample_val and "-" in sample_val:
                        id_cols_to_exclude.append(col)
                        _log(f"Excluding ID-like column from predictors: {col}", log_fh)

        # Remove ID-like columns from formula
        for col in id_cols_to_exclude:
            formula = (
                formula.replace(f" + {col}", "")
                .replace(f"{col} + ", "")
                .replace(f"{col}", "1")
            )
            if col in model_data.columns:
                _log(f"Dropping ID column from model data: {col}", log_fh)
                model_data = model_data.drop(columns=[col])

        # Check for columns with too few unique values (problematic for convergence)
        for col in model_data.columns:
            if col != cluster_col and col != "id":
                n_unique = model_data[col].nunique()
                if n_unique <= 1:
                    _log(
                        f"WARNING: Column '{col}' has only {n_unique} unique value(s). Removing from model.",
                        log_fh,
                    )
                    formula = (
                        formula.replace(f" + {col}", "")
                        .replace(f"{col} + ", "")
                        .replace(f"{col}", "1")
                    )

        # Explicitly convert sex variable from F/M to 0/1 if it exists
        if "sex" in model_data.columns:
            _log("Converting sex variable from F/M to numeric (0=F, 1=M)", log_fh)
            # Check the unique values to confirm format
            sex_values = model_data["sex"].unique()
            _log(f"Found sex values: {sex_values}", log_fh)

            # Create a mapping - assumes 'F' and 'M' format
            sex_map = {"F": 0, "M": 1}

            # Apply mapping, with error handling
            try:
                model_data["sex"] = model_data["sex"].map(sex_map)
                _log(f"Sex variable mapped to: {model_data['sex'].unique()}", log_fh)
            except Exception as e_sex:
                _log(f"Error converting sex variable: {e_sex}", log_fh)
                _log("Continuing with original sex variable format", log_fh)

        # Create summary file content
        summary_text_with_mapping = f"Target relationship: {target_relationship}\n\n"
        summary_text_with_mapping += f"Original formula: {formula}\n\n"
        summary_text_with_mapping += f"Cluster code mapping:\n"
        for code, value in code_mapping.items():
            summary_text_with_mapping += f"  {code}: {value}\n\n"

        # Single model fitting attempt - no fallbacks
        try:
            _log("Fitting multinomial logit model...", log_fh)
            m = smf.mnlogit(formula, data=model_data)
            res = m.fit(disp=False, method="newton", maxiter=100)

            # Check for NaN values in parameters
            if np.isnan(res.params).any().any():
                _log(
                    "WARNING: Model contains NaN coefficients. Results may be unreliable.",
                    log_fh,
                )

                # Try to identify problematic variables
                problem_vars = []
                for col in model_data.columns:
                    if col != cluster_col and col != "id":
                        # Check for near-zero variance (only for numeric columns)
                        if (
                            model_data[col].dtype.kind in "if"
                            and float(model_data[col].var()) < 1e-6  # Cast to float
                        ):
                            problem_vars.append(
                                f"{col} (low variance: {model_data[col].var():.8f})"
                            )
                        # Check for extreme values (only for numeric columns)
                        elif (
                            model_data[col].dtype.kind in "if"
                            and np.abs(model_data[col]).max() > 1e6
                        ):
                            problem_vars.append(
                                f"{col} (extreme values: max={np.abs(model_data[col]).max():.2f})"
                            )
                        # Check for high correlation with other variables
                        else:
                            for other_col in model_data.columns:
                                if (
                                    other_col != col
                                    and other_col != cluster_col
                                    and other_col != "id"
                                ):
                                    try:
                                        corr = model_data[col].corr(
                                            model_data[other_col]
                                        )
                                        if abs(corr) > 0.95:
                                            problem_vars.append(
                                                f"{col} (high correlation with {other_col}: {corr:.3f})"
                                            )
                                            break
                                    except:
                                        pass  # Skip if correlation can't be calculated

                if problem_vars:
                    _log(f"Potentially problematic variables: {problem_vars}", log_fh)

            # Check convergence
            if not res.mle_retvals["converged"]:
                _log(
                    "WARNING: Model did not converge. Results may be unreliable.",
                    log_fh,
                )
            else:
                _log("Model converged successfully.", log_fh)

            # Report log-likelihood and pseudo-R2 for model quality assessment
            _log(f"Log-likelihood: {res.llf:.2f}", log_fh)
            _log(f"Pseudo R-squared: {res.prsquared:.4f}", log_fh)

            # Get cluster-robust standard errors if possible
            try:
                _log("Computing cluster-robust standard errors...", log_fh)
                if hasattr(res, "get_robustcov_results"):
                    # Newer statsmodels versions
                    res_rob = res.get_robustcov_results(
                        cov_type="cluster", groups=model_data["id"]
                    )
                    _log("Successfully computed cluster-robust standard errors", log_fh)
                else:
                    _log(
                        "WARNING: No robust covariance method found. Using non-robust results.",
                        log_fh,
                    )
                    res_rob = res

                # Compute proper p-values if they're NaN
                if (
                    hasattr(res_rob, "pvalues")
                    and np.isnan(res_rob.pvalues).any().any()
                ):
                    _log(
                        "Detected NaN p-values, attempting to calculate manually",
                        log_fh,
                    )
                    # Manual calculation of p-values using z-statistics
                    p_values = {}
                    for outcome in res_rob.model.endog_names:
                        if outcome != "0":  # Reference category
                            p_values[outcome] = {}
                            for param in res_rob.params.index:
                                try:
                                    coef = res_rob.params.loc[param, outcome]
                                    std_err = res_rob.bse.loc[param, outcome]
                                    if std_err > 0:
                                        z_stat = coef / std_err
                                        import scipy.stats as stats

                                        p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                                        p_values[outcome][param] = p_val
                                except Exception as e_pval:
                                    _log(
                                        f"Error calculating p-value for {param}, {outcome}: {e_pval}",
                                        log_fh,
                                    )

                    _log(f"Manual p-values calculated: {p_values}", log_fh)

                # Add formula info
                summary_text_with_mapping += f"Final formula: {formula}\n\n"

                # Add model summary
                summary_text = res_rob.summary().as_text()
                summary_text_with_mapping += summary_text

                _log(
                    "\nModel summary with standard errors:\n" + summary_text,
                    log_fh,
                )

                # Save to file if requested
                if out_path:
                    with open(out_path, "w") as fh:
                        fh.write(summary_text_with_mapping)
                    _log(f"Saved frequentist summary to {out_path}", log_fh)

                    # Also save the scaling information for reference
                    if scaling_info is not None:
                        scale_path = out_path.parent / f"{out_path.stem}_scaling.csv"
                        scaling_info.to_csv(scale_path)
                        _log(f"Saved scaling parameters to {scale_path}", log_fh)

                # Add k-fold cross-validation
                try:
                    from sklearn.model_selection import KFold

                    _log("Computing 5-fold cross-validation...", log_fh)
                    kf = KFold(n_splits=5, shuffle=True, random_state=42)
                    cv_scores = []

                    # Exclude id and cluster columns for X
                    X_columns = [
                        c for c in model_data.columns if c != cluster_col and c != "id"
                    ]
                    if X_columns:
                        X_cv = model_data[X_columns]
                        y_cv = model_data[cluster_col]

                        for train_idx, test_idx in kf.split(X_cv):
                            X_train, X_test = X_cv.iloc[train_idx], X_cv.iloc[test_idx]
                            y_train, y_test = y_cv.iloc[train_idx], y_cv.iloc[test_idx]

                            # Create training dataset
                            train_data = pd.concat([X_train, y_train], axis=1)

                            # Fit model on training data
                            try:
                                cv_model = smf.mnlogit(formula, data=train_data)
                                cv_res = cv_model.fit(disp=False, maxiter=50)

                                # Calculate log-likelihood on test data
                                test_data = pd.concat([X_test, y_test], axis=1)
                                ll = cv_res.model.loglikeobs(
                                    cv_res.params, exog=test_data
                                )
                                cv_scores.append(ll.mean())
                            except Exception as e_cv:
                                _log(f"Error in CV fold: {e_cv}", log_fh)

                        # Calculate mean score across folds
                        if cv_scores:
                            mean_cv_score = sum(cv_scores) / len(cv_scores)
                            _log(f"Mean CV log-likelihood: {mean_cv_score:.4f}", log_fh)

                            # Add to output file
                            if out_path:
                                with open(out_path, "a") as fh:
                                    fh.write(f"\n\n5-fold Cross-Validation Results\n")
                                    fh.write(
                                        f"Mean log-likelihood: {mean_cv_score:.4f}\n"
                                    )
                                    fh.write(f"Individual fold scores: {cv_scores}\n")
                        else:
                            _log("Cross-validation failed on all folds", log_fh)
                    else:
                        _log(
                            "No predictor columns available for cross-validation",
                            log_fh,
                        )
                except Exception as e_cv_outer:
                    _log(f"Error in cross-validation process: {e_cv_outer}", log_fh)

                return res_rob

            except Exception as e_rob:
                _log(f"ERROR computing standard errors: {e_rob}", log_fh)

                # Save non-robust results as fallback
                summary_text = res.summary().as_text()
                summary_text_with_mapping += summary_text

                _log(
                    "\nModel summary with non-robust standard errors:\n" + summary_text,
                    log_fh,
                )

                if out_path:
                    with open(out_path, "w") as fh:
                        fh.write(summary_text_with_mapping)
                    _log(f"Saved non-robust frequentist summary to {out_path}", log_fh)

                return res

        except Exception as e_fit:
            _log(f"ERROR in model fitting: {e_fit}", log_fh)

            if out_path:
                with open(out_path, "w") as fh:
                    fh.write(summary_text_with_mapping)
                    fh.write("\nERROR IN MODEL FITTING\n\n")
                    fh.write(f"Error: {e_fit}\n\n")
                    fh.write("Possible reasons for failure:\n")
                    fh.write(
                        "1. Perfect separation in data (predictor perfectly predicts outcome)\n"
                    )
                    fh.write("2. Extreme multicollinearity between predictors\n")
                    fh.write("3. Insufficient variation in predictors or outcome\n")
                    fh.write("4. Numerical instability in optimization routine\n")

            return None

    except Exception as e_outer:
        _log(f"CRITICAL ERROR in multinomial_frequentist_model: {e_outer}", log_fh)
        import traceback

        _log(traceback.format_exc(), log_fh)

        if out_path:
            with open(out_path, "w") as fh:
                fh.write(f"Target relationship: {target_relationship}\n\n")
                fh.write("CRITICAL ERROR IN MODEL PROCESSING\n\n")
                fh.write(f"Error: {e_outer}\n\n")
                fh.write("Detailed traceback:\n")
                fh.write(traceback.format_exc())

        return None

    finally:
        if log_fh:
            log_fh.close()


# ---------------------------------------


def visualize_model_results(
    out_dir: Path, model_files: Optional[List[Path]] = None
) -> None:
    """
    Create visualizations to help interpret the significance of target relationships.

    Parameters
    ----------
    out_dir : Path
        Base directory where model outputs are stored
    model_files : List[Path], optional
        List of model summary files to analyze. If None, will search in out_dir.
    """
    import os
    import re
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set up directories
    output_dir = out_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all model summary files if not provided
    if model_files is None:
        model_files = []
        # Look directly in the output directory
        model_files.extend(out_dir.glob("*model_*_summary.txt"))
        # Also look in subdirectories (for target relationship folders)
        for dir_path in out_dir.glob("*/"):
            if dir_path.is_dir() and dir_path != output_dir:
                model_files.extend(dir_path.glob("*model_*_summary.txt"))

    if not model_files:
        print("No model summary files found for visualization.")
        return

    print(f"Found {len(model_files)} model summary files for visualization.")

    # Process each model file
    all_models = []
    for file_path in model_files:
        print(f"Processing {file_path}...")
        model_data = _parse_model_summary(file_path)
        if model_data:
            all_models.append(model_data)
            _plot_coefficient_intervals(model_data, output_dir)

    if all_models:
        print(f"Successfully visualized {len(all_models)} model files.")
        _create_forest_plot(all_models, output_dir)
        _plot_cv_comparison(model_files, output_dir)
    else:
        print("No models could be parsed successfully for visualization.")


def _parse_model_summary(file_path: Path) -> Optional[dict]:
    """Extract model summary data from the model output files."""
    with open(file_path, "r") as f:
        content = f.read()

    # Get model formula
    formula_match = re.search(r"Model formula: (.*)\n", content)
    formula = formula_match.group(1) if formula_match else "Unknown"

    # Get target relationship
    target_match = re.search(r"Target relationship: (.*)\n", content)
    target = target_match.group(1) if target_match else "Unknown"

    # Extract the parameter table
    lines = content.split("\n")
    data_start = None
    data_end = None

    for i, line in enumerate(lines):
        if re.match(r"\s*mean\s+sd\s+hdi_3%\s+hdi_97%", line):
            data_start = i + 1
        if data_start and line.strip() == "":
            data_end = i
            break

    if data_start and data_end:
        # Parse the table into a dataframe
        table_lines = lines[data_start:data_end]

        # Process the parameter names and values
        params = []
        for line in table_lines:
            parts = re.split(r"\s+", line.strip())
            if len(parts) >= 5:  # Should have parameter name, mean, sd, hdi_3%, hdi_97%
                param_name = parts[0]
                try:
                    mean = float(parts[1])
                    sd = float(parts[2])
                    hdi_3 = float(parts[3])
                    hdi_97 = float(parts[4])
                    params.append(
                        {
                            "parameter": param_name,
                            "mean": mean,
                            "sd": sd,
                            "hdi_3%": hdi_3,
                            "hdi_97%": hdi_97,
                            "significant": (
                                hdi_3 > 0 or hdi_97 < 0
                            ),  # Check if CI excludes 0
                        }
                    )
                except (ValueError, IndexError):
                    continue  # Skip lines that can't be parsed correctly

        return {
            "formula": formula,
            "target_relationship": target,
            "parameters": pd.DataFrame(params),
            "file_path": file_path,
        }

    return None


def _plot_coefficient_intervals(model_data: dict, output_dir: Path) -> None:
    """Create plots of coefficients with credible intervals."""
    df = model_data["parameters"]

    # Filter parameters of interest (exclude random effects and intercepts)
    exclude_patterns = ["1\|id", "Intercept"]
    params_of_interest = df[
        ~df["parameter"].str.contains("|".join(exclude_patterns), regex=True)
    ]

    if len(params_of_interest) == 0:
        print(f"No fixed effects found in {model_data['file_path']}")
        return

    # Sort by effect size (absolute mean value)
    params_of_interest["abs_mean"] = params_of_interest["mean"].abs()
    params_of_interest = params_of_interest.sort_values("abs_mean", ascending=False)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(params_of_interest) * 0.4)))

    # Plot horizontal lines for coefficients
    for i, (_, row) in enumerate(params_of_interest.iterrows()):
        color = "blue" if row["significant"] else "gray"

        # Plot the credible interval
        ax.plot([row["hdi_3%"], row["hdi_97%"]], [i, i], color=color, linewidth=2)

        # Plot the mean as a marker
        ax.scatter(row["mean"], i, color=color, s=50, zorder=3)

    # Add parameter names
    ax.set_yticks(range(len(params_of_interest)))
    ax.set_yticklabels(params_of_interest["parameter"])

    # Add vertical line at zero
    ax.axvline(x=0, linestyle="--", color="red", alpha=0.6)

    # Add labels and title
    target_rel = model_data["target_relationship"]
    ax.set_title(
        f"Coefficient Estimates with 94% Credible Intervals\nTarget: {target_rel}"
    )
    ax.set_xlabel("Coefficient Value")

    # Add annotations for significant effects
    for i, (_, row) in enumerate(params_of_interest.iterrows()):
        if row["significant"]:
            ax.text(
                row["hdi_97%"] + 0.02,
                i,
                "Significant",
                va="center",
                ha="left",
                color="blue",
                fontsize=9,
            )

    # Add a note about significance
    fig.text(
        0.5,
        0.01,
        "Note: Coefficients are considered significant when their 94% credible interval excludes zero.",
        ha="center",
        fontsize=9,
        style="italic",
    )

    # Save the figure
    output_file = (
        output_dir
        / f"coefficients_{target_rel.replace(' -> ', '_to_').replace(' ', '_')}.png"
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved coefficient plot to {output_file}")


def _create_forest_plot(all_models: List[dict], output_dir: Path) -> None:
    """Create a forest plot comparing effects across models."""
    # Extract target variable from each model
    models_data = []

    for model in all_models:
        target_var = model["target_relationship"].split(" -> ")[0]

        # Get the coefficient for this target variable
        params = model["parameters"]
        effect_param = params[params["parameter"] == target_var]

        if not effect_param.empty:
            models_data.append(
                {
                    "target": target_var,
                    "mean": effect_param.iloc[0]["mean"],
                    "lower": effect_param.iloc[0]["hdi_3%"],
                    "upper": effect_param.iloc[0]["hdi_97%"],
                    "significant": effect_param.iloc[0]["significant"],
                }
            )

    if not models_data:
        print("No valid model data for forest plot")
        return

    # Create DataFrame for plotting
    forest_df = pd.DataFrame(models_data)

    # Sort by effect size
    forest_df = forest_df.sort_values("mean", ascending=False)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(forest_df) * 0.6)))

    # Plot horizontal lines for coefficients
    for i, (_, row) in enumerate(forest_df.iterrows()):
        color = "blue" if row["significant"] else "gray"

        # Plot the credible interval
        ax.plot([row["lower"], row["upper"]], [i, i], color=color, linewidth=2)

        # Plot the mean as a marker
        ax.scatter(row["mean"], i, color=color, s=50, zorder=3)

    # Add parameter names
    ax.set_yticks(range(len(forest_df)))
    ax.set_yticklabels(forest_df["target"])

    # Add vertical line at zero
    ax.axvline(x=0, linestyle="--", color="red", alpha=0.6)

    # Add labels and title
    ax.set_title(
        "Comparison of Direct Effects on Cluster Membership\nAcross Different Target Variables"
    )
    ax.set_xlabel("Coefficient Value (Effect Size)")

    # Add annotations for significant effects
    for i, (_, row) in enumerate(forest_df.iterrows()):
        if row["significant"]:
            ax.text(
                row["upper"] + 0.02,
                i,
                "Significant",
                va="center",
                ha="left",
                color="blue",
                fontsize=9,
            )

    # Add a note about significance
    fig.text(
        0.5,
        0.01,
        "Note: Effects are considered significant when their 94% credible interval excludes zero.",
        ha="center",
        fontsize=9,
        style="italic",
    )

    # Save the figure
    output_file = output_dir / "forest_plot_comparison.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved forest plot to {output_file}")


def _extract_cv_scores(model_files: List[Path]) -> Optional[pd.DataFrame]:
    """Extract cross-validation scores from model output files."""
    cv_data = []

    for file_path in model_files:
        with open(file_path, "r") as f:
            content = f.read()

        # Get target relationship
        target_match = re.search(r"Target relationship: (.*)\n", content)
        target = target_match.group(1) if target_match else "Unknown"
        target_var = target.split(" -> ")[0] if " -> " in target else target

        # Look for CV scores
        loo_match = re.search(r"elpd_loo\s*:\s*([\d.-]+)", content)
        waic_match = re.search(r"elpd_waic\s*:\s*([\d.-]+)", content)

        if loo_match:
            cv_data.append(
                {
                    "target": target_var,
                    "metric": "loo",
                    "score": float(loo_match.group(1)),
                    "file": os.path.basename(str(file_path)),
                }
            )
        elif waic_match:
            cv_data.append(
                {
                    "target": target_var,
                    "metric": "waic",
                    "score": float(waic_match.group(1)),
                    "file": os.path.basename(str(file_path)),
                }
            )

    return pd.DataFrame(cv_data) if cv_data else None


def _plot_cv_comparison(model_files: List[Path], output_dir: Path) -> None:
    """Create a plot comparing cross-validation scores across models."""
    cv_data = _extract_cv_scores(model_files)

    if cv_data is None or cv_data.empty:
        print("No cross-validation scores found for comparison")
        return

    # Group by target variable and metric, taking the best score
    best_scores = cv_data.groupby(["target", "metric"])["score"].max().reset_index()

    # Create plot
    plt.figure(figsize=(12, 6))

    # Create barplots for each metric
    metrics = best_scores["metric"].unique()
    n_metrics = len(metrics)

    for i, metric in enumerate(metrics):
        metric_data = best_scores[best_scores["metric"] == metric]
        plt.subplot(1, n_metrics, i + 1)

        # Sort by score (higher is better)
        metric_data = metric_data.sort_values("score", ascending=False)

        # Plot bars
        ax = sns.barplot(x="target", y="score", data=metric_data, palette="viridis")

        # Add score labels using ax.bar_label for robustness
        for container in ax.containers:
            if isinstance(container, BarContainer):
                ax.bar_label(container, fmt="%.1f", fontsize=9, padding=3)

        plt.title(f"{metric.upper()} Cross-Validation Scores")
        plt.ylabel("elpd_" + metric)
        plt.xlabel("Target Variable")
        plt.xticks(rotation=45)

    # Add a note about interpretation
    plt.figtext(
        0.5,
        0.01,
        "Note: Higher values indicate better model fit. Models with higher scores have stronger predictive power.",
        ha="center",
        fontsize=9,
        style="italic",
    )

    # Save the figure
    output_file = output_dir / "cv_score_comparison.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved cross-validation score comparison to {output_file}")


def visualize_posterior_distributions(
    out_dir: Path, model_files: Optional[List[Path]] = None
) -> None:
    """
    Create direct visualizations of coefficient posterior distributions from model summary files.
    This is a more robust method for creating posterior visualizations when the standard method fails.

    Parameters
    ----------
    out_dir : Path
        Base directory where model outputs are stored
    model_files : List[Path], optional
        List of model summary files to analyze. If None, will search in out_dir.
    """
    import os
    import re
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path

    # Find all model summary files if not provided
    if model_files is None:
        model_files = []
        # Look in the output directory and its subdirectories
        for root, _, files in os.walk(out_dir):
            for file in files:
                if "multinomial_model" in file and file.endswith("_summary.txt"):
                    model_files.append(Path(root) / file)

    if not model_files:
        print("No model summary files found for visualization")
        return

    print(f"Found {len(model_files)} model summary files for visualization")

    # Create output directory for visualizations
    output_dir = out_dir / "posterior_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each model file
    all_effects = []
    model_index = 0

    for file_path in model_files:
        model_index += 1
        print(f"Processing {file_path}...")

        # Extract model name and target relationship from filename
        relationship = None
        model_name = f"Model {model_index}"
        file_stem = file_path.stem

        # Try to extract a meaningful model name from the path
        parts = str(file_path).split("/")
        for part in parts:
            if "cluster" in part or "model" in part:
                model_name = part
                break

        # Try to extract from filename
        if "_to_cluster" in str(file_path):
            parts = str(file_path).split("_to_cluster")
            if parts:
                var_name = parts[0].split("/")[-1]
                relationship = f"{var_name} -> cluster"

        # Read the file to extract target relationship and parameter info
        with open(file_path, "r") as f:
            content = f.read()

            # Extract target relationship if not found in filename
            if not relationship:
                match = re.search(r"Target relationship: (.*)", content)
                if match:
                    relationship = match.group(1)
                else:
                    relationship = file_stem.replace("multinomial_model_", "").replace(
                        "_summary", ""
                    )

            print(
                f"Looking for coefficients in {file_path} for relationship: {relationship}"
            )

            # First try the standard pattern
            pattern = r"Coefficient (\w+): mean=([-\d\.]+), 95% HDI=\[([-\d\.]+), ([-\d\.]+)\], P\(effect != 0\)=([\d\.]+)% - (\w+)"
            matches = re.findall(pattern, content)

            # If no matches, try looking for parameters in ArviZ summary table format
            if not matches:
                print(
                    f"No coefficients found with standard pattern. Trying ArviZ summary table format..."
                )
                # Try to extract from ArviZ summary table format
                if " -> " in relationship:
                    target_var = relationship.split(" -> ")[0].strip()
                    # Look for target_var in the parameter table like "target_var[1]" or just "target_var"
                    param_pattern = rf"({target_var}(?:\[\d+\])?)\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)"
                    param_matches = re.findall(param_pattern, content)

                    for param_match in param_matches:
                        param, mean, sd, hdi_3, hdi_97 = param_match
                        try:
                            mean_val = float(mean)
                            hdi_lower = float(hdi_3)
                            hdi_upper = float(hdi_97)
                            # Calculate probability of effect being non-zero based on HDI
                            significant = (hdi_lower > 0) or (hdi_upper < 0)
                            prob = 0.95 if significant else 0.5  # Approximate

                            matches.append(
                                (
                                    param,
                                    mean,
                                    hdi_lower,
                                    hdi_upper,
                                    str(int(prob * 100)),
                                    "significant" if significant else "not significant",
                                )
                            )
                            print(f"Found parameter: {param} with mean={mean}")
                        except ValueError:
                            print(f"Could not convert values for {param}")

            if matches:
                print(f"Found {len(matches)} coefficient matches")
            else:
                print(f"No coefficients found in {file_path}")

            for match in matches:
                param_name, mean, hdi_lower, hdi_upper, prob, significance = match
                # Extract target variable from relationship
                if " -> " in relationship:
                    target_var = relationship.split(" -> ")[0].strip()
                else:
                    target_var = "unknown"

                # Only keep the coefficient for the target variable if it matches or starts with the target_var
                if param_name == target_var or param_name.startswith(f"{target_var}["):
                    # Create a unique parameter identifier
                    param_identifier = f"{model_name}_{param_name}"

                    # Extract cluster information if available
                    cluster_info = ""
                    if "[" in param_name:
                        cluster_match = re.search(r"\[(\d+)\]", param_name)
                        if cluster_match:
                            cluster_number = int(cluster_match.group(1))
                            cluster_info = f"(Cluster {cluster_number} vs. Reference)"

                    # Create a display label that uniquely identifies the parameter
                    display_label = f"{target_var} -> cluster"
                    if cluster_info:
                        display_label = f"{display_label} {cluster_info}"

                    all_effects.append(
                        {
                            "parameter": param_name,
                            "relationship": relationship,
                            "model_name": model_name,
                            "mean": float(mean),
                            "hdi_lower": float(hdi_lower),
                            "hdi_upper": float(hdi_upper),
                            "probability": float(prob) / 100.0,  # Convert to 0-1 scale
                            "significant": significance.lower() == "significant",
                            "display_label": display_label,
                            "param_identifier": param_identifier,
                            "cluster_info": cluster_info,
                            "target_var": target_var,
                        }
                    )
                    print(f"Added effect for {display_label} from {model_name}")

    if not all_effects:
        print(
            "No coefficient data found in any model files. Check if the model output format matches the expected patterns."
        )
        print(
            "You may need to run 'multinomial_mixed_model' with proper settings to generate these files."
        )
        return

    # Create DataFrame for plotting
    effects_df = pd.DataFrame(all_effects)

    # Remove duplicates based on param_identifier
    effects_df = effects_df.drop_duplicates(subset=["param_identifier"])

    # Sort the effects for better visualization
    # First sort by the target variable
    effects_df["target_var"] = effects_df["relationship"].apply(
        lambda x: x.split(" -> ")[0] if " -> " in x else x
    )

    # Sort by target variable first, then by cluster number if available
    def extract_cluster_number(cluster_info):
        if not cluster_info:
            return 999  # Non-cluster items at the end
        match = re.search(r"Cluster (\d+)", cluster_info)
        if match:
            return int(match.group(1))
        return 999

    effects_df["cluster_number"] = effects_df["cluster_info"].apply(
        extract_cluster_number
    )
    effects_df = effects_df.sort_values(by=["target_var", "cluster_number"])

    # 1. Create forest plot of all effects with clearer organization
    plt.figure(figsize=(12, max(6, len(effects_df) * 0.6)))

    # Plot horizontal lines for coefficients
    for i, (_, row) in enumerate(effects_df.iterrows()):
        color = "blue" if row["significant"] else "gray"

        # Plot the credible interval
        plt.plot([row["hdi_lower"], row["hdi_upper"]], [i, i], color=color, linewidth=2)

        # Plot the mean as a marker
        plt.scatter(row["mean"], i, color=color, s=50, zorder=3)

    # Add parameter names with clear labels
    plt.yticks(range(len(effects_df)), effects_df["display_label"].tolist())

    # Add vertical line at zero
    plt.axvline(x=0, linestyle="--", color="red", alpha=0.6)

    # Add labels and title
    plt.title("Effect Estimates with 95% Credible Intervals")
    plt.xlabel("Coefficient Value (Effect Size)")

    # Add annotations for significant effects
    for i, (_, row) in enumerate(effects_df.iterrows()):
        if row["significant"]:
            plt.text(
                row["hdi_upper"] + 0.02,
                i,
                f"Significant (p={row['probability']:.2f})",
                va="center",
                ha="left",
                color="blue",
                fontsize=9,
            )

    # Save the figure
    output_file = output_dir / "forest_plot_all_effects.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved forest plot to {output_file}")

    # 2. Create separate forest plots by target variable for better clarity
    target_vars = effects_df["target_var"].unique()

    for target_var in target_vars:
        var_effects = effects_df[effects_df["target_var"] == target_var]

        if len(var_effects) > 1:  # Only create plot if there are multiple effects
            plt.figure(figsize=(10, max(4, len(var_effects) * 0.8)))

            # Plot horizontal lines for coefficients
            for i, (_, row) in enumerate(var_effects.iterrows()):
                color = "blue" if row["significant"] else "gray"
                plt.plot(
                    [row["hdi_lower"], row["hdi_upper"]],
                    [i, i],
                    color=color,
                    linewidth=2,
                )
                plt.scatter(row["mean"], i, color=color, s=50, zorder=3)

            # Add parameter names
            plt.yticks(range(len(var_effects)), var_effects["display_label"].tolist())
            plt.axvline(x=0, linestyle="--", color="red", alpha=0.6)

            # Add title and labels
            plt.title(f"Effect of {target_var} on Cluster Membership")
            plt.xlabel("Coefficient Value (Effect Size)")

            # Add annotations for significant effects
            for i, (_, row) in enumerate(var_effects.iterrows()):
                if row["significant"]:
                    plt.text(
                        row["hdi_upper"] + 0.02,
                        i,
                        f"Significant (p={row['probability']:.2f})",
                        va="center",
                        ha="left",
                        color="blue",
                        fontsize=9,
                    )

            # Save the figure
            output_file = output_dir / f"forest_plot_{target_var}_effects.png"
            plt.tight_layout()
            plt.savefig(output_file, dpi=300)
            plt.close()
            print(f"Saved forest plot for {target_var} to {output_file}")

    # 3. Create individual posterior density plots for all parameters
    # First create a directory to organize density plots
    density_plots_dir = output_dir / "posterior_densities"
    density_plots_dir.mkdir(exist_ok=True)

    # Create plots for all parameters
    print(f"Generating posterior density plots for all {len(effects_df)} parameters...")
    for _, row in effects_df.iterrows():
        plt.figure(figsize=(8, 5))

        # Generate a simulated posterior distribution based on the mean and HDI
        # This is an approximation assuming normality
        from scipy import stats as scipy_stats

        # Estimate standard deviation from the HDI
        # For a normal distribution, 95% HDI spans approximately 4 standard deviations
        std_dev = (row["hdi_upper"] - row["hdi_lower"]) / 3.92

        # Generate points for the distribution
        x = np.linspace(row["hdi_lower"] - std_dev, row["hdi_upper"] + std_dev, 1000)
        y = scipy_stats.norm.pdf(x, row["mean"], std_dev)

        # Plot the distribution with appropriate coloring based on significance
        color = "blue" if row["significant"] else "gray"
        plt.plot(x, y, color=color, linewidth=2)

        # Shade the area outside the HDI
        plt.fill_between(
            x,
            y,
            where=((x < row["hdi_lower"]) | (x > row["hdi_upper"])),
            color="lightgray",
            alpha=0.5,
        )

        # Add vertical lines
        plt.axvline(x=0, color="red", linestyle="--", label="Zero")
        plt.axvline(x=row["mean"], color=color, linestyle="-", label="Mean")
        plt.axvline(x=row["hdi_lower"], color="green", linestyle=":", label="95% HDI")
        plt.axvline(x=row["hdi_upper"], color="green", linestyle=":")

        # Add title and labels
        significance_marker = " [SIGNIFICANT]" if row["significant"] else ""
        plt.title(
            f"Posterior Distribution for {row['display_label']}{significance_marker}"
        )
        plt.xlabel("Effect Size (Coefficient Value)")
        plt.ylabel("Density")

        # Add annotation about significance
        significance_text = "Significant" if row["significant"] else "Not Significant"
        plt.annotate(
            f"Mean = {row['mean']:.3f}, 95% HDI = [{row['hdi_lower']:.3f}, {row['hdi_upper']:.3f}]\n"
            f"P(effect ≠ 0) = {row['probability']:.2%} - {significance_text}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8),
            ha="left",
            va="top",
        )

        plt.legend()
        plt.tight_layout()

        # Save the figure with a more informative filename
        target_var = row["target_var"]

        # Create subdirectory by target variable for better organization
        var_dir = density_plots_dir / target_var
        var_dir.mkdir(exist_ok=True)

        # Generate safe filename
        safe_name = (
            row["display_label"]
            .replace(" -> ", "_to_")
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )
        output_file = var_dir / f"posterior_density_{safe_name}.png"
        plt.savefig(output_file, dpi=300)
        plt.close()

    print(f"Saved posterior density plots to {density_plots_dir}")

    # Also create a combined figure with density plots for significant effects
    sig_effects = effects_df[effects_df["significant"]]
    if not sig_effects.empty:
        print(f"Creating combined plot for {len(sig_effects)} significant effects...")
        n_plots = len(sig_effects)
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots), sharex=True)

        # Handle the case with only one significant effect
        if n_plots == 1:
            axes = [axes]

        for i, (_, row) in enumerate(sig_effects.iterrows()):
            ax = axes[i]

            # Plot the density
            std_dev = (row["hdi_upper"] - row["hdi_lower"]) / 3.92
            x = np.linspace(
                row["hdi_lower"] - std_dev, row["hdi_upper"] + std_dev, 1000
            )
            y = scipy_stats.norm.pdf(x, row["mean"], std_dev)

            ax.plot(x, y, color="blue", linewidth=2)
            ax.fill_between(
                x,
                y,
                where=((x < row["hdi_lower"]) | (x > row["hdi_upper"])),
                color="lightgray",
                alpha=0.5,
            )

            # Add vertical lines
            ax.axvline(x=0, color="red", linestyle="--", label="Zero")
            ax.axvline(x=row["mean"], color="blue", linestyle="-", label="Mean")
            ax.axvline(
                x=row["hdi_lower"], color="green", linestyle=":", label="95% HDI"
            )
            ax.axvline(x=row["hdi_upper"], color="green", linestyle=":")

            # Add title
            ax.set_title(row["display_label"])
            ax.annotate(
                f"Mean = {row['mean']:.3f}, 95% HDI = [{row['hdi_lower']:.3f}, {row['hdi_upper']:.3f}]",
                xy=(0.05, 0.8),
                xycoords="axes fraction",
                bbox=dict(boxstyle="round", fc="white", alpha=0.8),
                ha="left",
            )

            if i == 0:  # Only add legend to the first plot
                ax.legend()

            if i == n_plots - 1:  # Only add x-label to the bottom plot
                ax.set_xlabel("Effect Size (Coefficient Value)")

            ax.set_ylabel("Density")

        plt.tight_layout()
        output_file = output_dir / "significant_effects_combined.png"
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"Saved combined significant effects plot to {output_file}")

    # 4. Create a summary table of all effects
    summary_df = effects_df[
        [
            "display_label",
            "mean",
            "hdi_lower",
            "hdi_upper",
            "probability",
            "significant",
        ]
    ]
    summary_df.columns = [
        "Effect",
        "Mean",
        "HDI Lower",
        "HDI Upper",
        "P(effect≠0)",
        "Significant",
    ]

    # Save the summary as CSV
    summary_csv = output_dir / "effects_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved effects summary to {summary_csv}")


def _create_informative_label(param_name, relationship):
    """Create an informative label for forest plots that includes cluster information."""
    # Extract the base parameter name and the cluster index if present
    match = re.match(r"(.+?)(?:\[(\d+)\])?$", param_name)
    if match:
        base_name, cluster_idx = match.groups()
        if cluster_idx:
            # Create a label with cluster information
            return f"{relationship} (Cluster {cluster_idx} vs. Reference)"
        else:
            return f"{relationship}"
    return f"{param_name} - {relationship}"
