"""
Causal inference module for sleep data analysis.

This module contains functions for causal graph creation and
identifying confounders using DoWhy.
"""

from typing import List, Optional, Union, Dict, Any
from pathlib import Path

import pandas as pd
import numpy as np
import networkx as nx
from dowhy import CausalModel

# Add graphviz import with try-except to make it optional
try:
    import graphviz

    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    graphviz = None  # Explicitly set graphviz to None if not available

# Add matplotlib import with try-except to make it optional
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None  # Explicitly set plt to None if not available


def create_sleep_causal_graph() -> nx.DiGraph:
    """
    Create a causal graph for sleep-mental health relationships.

    Notes
    -----
    * The graph includes both 'chronotype_desync' (raw desynchronization) and
      'abs_chronotype_desync' (absolute value of desynchronization).
    * 'episode_hr_mean' is used (there is no separate 'resting_hr' variable in the data).

    Returns
    -------
    nx.DiGraph
        NetworkX directed graph representing the causal model
    """
    # Create a new directed graph
    G = nx.DiGraph()

    # Add edges representing causal relationships

    # Demographics influence sleep and chronotype
    G.add_edge("age", "cluster")
    G.add_edge("age", "patient_chronotype")
    G.add_edge("age", "nadir_time_pct")
    G.add_edge("age", "se")

    # Sex influences sleep and chronotype
    G.add_edge("sex", "cluster")
    G.add_edge("sex", "patient_chronotype")
    G.add_edge("sex", "nadir_time_pct")

    # Chronotype influences sleep patterns
    G.add_edge("patient_chronotype", "cluster")
    G.add_edge("patient_chronotype", "chronotype_desync")
    G.add_edge("patient_chronotype", "abs_chronotype_desync")
    G.add_edge("patient_chronotype", "nadir_time_pct")

    # Add direct effect of chronotype desynchronization to cluster
    G.add_edge("chronotype_desync", "cluster")
    G.add_edge("chronotype_desync", "se")

    # Add direct effect of absolute chronotype desynchronization to cluster
    G.add_edge("abs_chronotype_desync", "cluster")
    G.add_edge("abs_chronotype_desync", "se")

    # Sleep parameters affect cluster membership and each other
    G.add_edge("nadir_time_pct", "cluster")
    G.add_edge("nadir_hr", "cluster")
    G.add_edge("se", "cluster")
    G.add_edge("waso", "cluster")
    G.add_edge("waso", "se")
    G.add_edge("waso", "sfi")
    G.add_edge("episode_hr_mean", "cluster")
    G.add_edge("sfi", "cluster")
    G.add_edge("sfi", "se")

    # Sleep regularity affects everything
    G.add_edge("sri", "cluster")
    G.add_edge("sri", "patient_chronotype")
    G.add_edge("sri", "chronotype_desync")
    G.add_edge("sri", "abs_chronotype_desync")
    G.add_edge("sri", "se")
    G.add_edge("sri", "nadir_time_pct")

    # Physical parameters affect HR and sleep
    # HR affects cluster membership and other HR measures
    G.add_edge("episode_hr_mean", "cluster")
    G.add_edge("episode_hr_mean", "nadir_hr")

    return G


def analyze_causal_paths(
    treatment: str, outcome: str, graph: nx.DiGraph
) -> Dict[str, Any]:
    """
    Analyze direct and indirect causal paths between treatment and outcome.
    Identifies whether direct effects exist, and finds potential mediators.

    Parameters
    ----------
    treatment : str
        The treatment/exposure variable
    outcome : str
        The outcome variable
    graph : nx.DiGraph
        The causal graph as a NetworkX directed graph

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'has_direct_effect': Boolean, whether a direct effect exists
        - 'has_indirect_effect': Boolean, whether indirect effects exist
        - 'mediators': List of mediator variables
        - 'all_paths': List of all paths from treatment to outcome
        - 'effect_type': String describing the type of effect (direct, indirect, or mixed)
    """
    # Check if both nodes exist in the graph
    if treatment not in graph.nodes or outcome not in graph.nodes:
        return {
            "has_direct_effect": False,
            "has_indirect_effect": False,
            "mediators": [],
            "all_paths": [],
            "effect_type": "none",
            "error": f"One or both nodes ({treatment}, {outcome}) not in graph",
        }

    # Check if there's a direct edge (direct effect)
    has_direct_effect = graph.has_edge(treatment, outcome)

    # Find all simple paths from treatment to outcome
    try:
        all_paths = list(nx.all_simple_paths(graph, treatment, outcome))
    except nx.NetworkXNoPath:
        all_paths = []

    # Filter to find indirect paths (length > 1 meaning they go through other nodes)
    indirect_paths = [path for path in all_paths if len(path) > 2]
    has_indirect_effect = len(indirect_paths) > 0

    # Identify mediators (nodes that appear in indirect paths)
    all_mediators = set()
    for path in indirect_paths:
        # Mediators are all nodes in the path except the first (treatment) and last (outcome)
        mediators_in_path = path[1:-1]
        all_mediators.update(mediators_in_path)

    # Sort mediators for consistent output
    mediators = sorted(list(all_mediators))

    # Determine the type of effect
    if has_direct_effect and has_indirect_effect:
        effect_type = "mixed (both direct and indirect effects)"
    elif has_direct_effect:
        effect_type = "direct effect only"
    elif has_indirect_effect:
        effect_type = "indirect effect only"
    else:
        effect_type = "no effect"

    return {
        "has_direct_effect": has_direct_effect,
        "has_indirect_effect": has_indirect_effect,
        "mediators": mediators,
        "all_paths": all_paths,
        "effect_type": effect_type,
    }


def get_manual_confounders() -> Dict[str, List[str]]:
    """
    Create a dictionary of manual confounders for causal relationships
    based on the DAG structure.

    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping causal relationships to their confounders
    """
    # Create the graph to validate against
    G = create_sleep_causal_graph()

    # Manual confounder identification based on the DAG structure
    manual_confounders = {
        "patient_chronotype -> cluster": ["age", "sex", "sri"],
        "age -> cluster": ["sex"],
        "sex -> cluster": ["age"],
        "chronotype_desync -> cluster": ["patient_chronotype", "sri", "age", "sex"],
        "abs_chronotype_desync -> cluster": ["patient_chronotype", "sri", "age", "sex"],
        "nadir_time_pct -> cluster": ["patient_chronotype", "age", "sex", "sri"],
        "nadir_hr -> cluster": ["episode_hr_mean"],
        "se -> cluster": [
            "chronotype_desync",
            "abs_chronotype_desync",
            "waso",
            "sri",
            "sfi",
            "age",
        ],
        "waso -> cluster": ["se", "sfi"],
        "sfi -> cluster": ["waso", "se"],
        "episode_hr_mean -> cluster": ["nadir_hr"],
        "sri -> cluster": [
            "patient_chronotype",
            "age",
            "sex",
            "chronotype_desync",
            "abs_chronotype_desync",
        ],
    }

    # Validate each relationship against the DAG
    for rel, confounders in manual_confounders.items():
        treatment, outcome = rel.split(" -> ")

        # Check that treatment and outcome nodes exist in the graph
        if treatment not in G.nodes:
            print(f"WARNING: Treatment '{treatment}' not found in graph nodes")
        if outcome not in G.nodes:
            print(f"WARNING: Outcome '{outcome}' not found in graph nodes")

        # Check that there's a direct edge between treatment and outcome
        if not G.has_edge(treatment, outcome):
            print(
                f"WARNING: No direct edge from '{treatment}' to '{outcome}' in the graph"
            )

        # Verify confounders are actual nodes in the graph
        for confounder in confounders:
            if confounder not in G.nodes:
                print(f"WARNING: Confounder '{confounder}' not found in graph nodes")

    return manual_confounders


def identify_confounders(
    df: pd.DataFrame,
    target_relationship: str = "patient_chronotype -> cluster",
    output_dir: Optional[Path] = None,
) -> List[str]:
    """
    Use DoWhy to identify confounders based on backdoor criterion.
    If DoWhy returns an empty set, falls back to manual identification.
    Also analyzes and logs information about direct/indirect effects and mediators.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data (needed for DoWhy)
    target_relationship : str
        The causal relationship of interest in format "treatment -> outcome"
        Examples: "patient_chronotype -> cluster", "nadir_time_pct -> cluster"
    output_dir : Optional[Path]
        Directory to save the causal analysis log file. If None, no log file is created.

    Returns
    -------
    List[str]
        List of variables that should be adjusted for (confounders)
    """
    # Get manual confounders dictionary
    manual_confounders = get_manual_confounders()

    # Create a causal graph
    graph = create_sleep_causal_graph()

    # Parse the relationship
    parts = target_relationship.split(" -> ")
    if len(parts) != 2:
        print(f"Invalid relationship format: {target_relationship}. Expected 'X -> Y'")
        return ["age", "sex"]  # Default confounders

    treatment, outcome = parts

    # Analyze the causal paths for this relationship
    paths_analysis = analyze_causal_paths(treatment, outcome, graph)

    # Get manual confounders for this relationship if available
    manual_conf_list = []
    has_manual_confounders = target_relationship in manual_confounders
    if has_manual_confounders:
        manual_conf_list = [
            c for c in manual_confounders[target_relationship] if c in df.columns
        ]
        print(f"Manual confounders for {target_relationship}: {manual_conf_list}")

    # Initialize the causal log file path variable
    causal_log_file = None

    # Create a log file for the causal paths analysis if output_dir is provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create a log file with details about the causal paths
        causal_log_file = output_dir / f"causal_analysis_{treatment}_to_{outcome}.txt"

        with open(causal_log_file, "w") as f:
            f.write(f"CAUSAL ANALYSIS FOR: {target_relationship}\n")
            f.write("=" * 50 + "\n\n")

            # Write path analysis results
            f.write("DIRECT AND INDIRECT EFFECTS:\n")
            f.write(f"Effect type: {paths_analysis['effect_type']}\n")
            f.write(f"Has direct effect: {paths_analysis['has_direct_effect']}\n")
            f.write(f"Has indirect effect: {paths_analysis['has_indirect_effect']}\n\n")

            # Write mediator information if there are any
            if paths_analysis["mediators"]:
                f.write("MEDIATORS (Variables in indirect paths):\n")
                for mediator in paths_analysis["mediators"]:
                    f.write(f"- {mediator}\n")
                f.write("\n")
            else:
                f.write("NO MEDIATORS IDENTIFIED\n\n")

            # Write all paths from treatment to outcome
            f.write("ALL CAUSAL PATHS:\n")
            if paths_analysis["all_paths"]:
                for i, path in enumerate(paths_analysis["all_paths"]):
                    f.write(f"Path {i+1}: {' -> '.join(path)}\n")
            else:
                f.write("No paths found from treatment to outcome.\n")
            f.write("\n")

            # Document the manual confounders approach
            f.write("MANUAL CONFOUNDERS APPROACH:\n")
            if has_manual_confounders:
                f.write(
                    f"Manual confounders available for {target_relationship}: {manual_confounders[target_relationship]}\n"
                )
                f.write(f"Manual confounders present in dataset: {manual_conf_list}\n")
            else:
                f.write(f"No manual confounders defined for {target_relationship}\n")
            f.write("\n")

            # Note about path-specific effects
            f.write("NOTE ON ESTIMATION:\n")
            if (
                paths_analysis["has_direct_effect"]
                and paths_analysis["has_indirect_effect"]
            ):
                f.write(
                    "This relationship has both direct and indirect effects. Consider:\n"
                )
                f.write("1. Total effect estimation (adjust for confounders only)\n")
                f.write(
                    "2. Direct effect estimation (adjust for confounders and mediators)\n"
                )
                f.write("3. Mediation analysis to decompose direct/indirect effects\n")
            elif paths_analysis["has_direct_effect"]:
                f.write(
                    "This relationship has only a direct effect. Standard adjustment for confounders is appropriate.\n"
                )
            elif paths_analysis["has_indirect_effect"]:
                f.write(
                    "This relationship has only indirect effects through mediators. Consider mediation analysis.\n"
                )
            else:
                f.write(
                    "No causal relationship found in the DAG between these variables.\n"
                )

            print(f"Causal path analysis saved to {causal_log_file}")

    try:
        # Create a clean copy of the dataframe for causal modeling
        df_causal = df.copy()

        # Log initial shapes
        print(
            f"Initially, DataFrame has {len(df_causal)} rows and {len(df_causal.columns)} columns"
        )

        # 1. Remove any array/object columns except strings
        obj_cols = []
        for col in df_causal.columns:
            if df_causal[col].dtype == "object":
                # Check first non-null value
                first_valid_idx = df_causal[col].first_valid_index()
                if first_valid_idx is not None:
                    # Use safe indexing to avoid linter errors
                    first_val = df_causal[col].iloc[
                        df_causal.index.get_indexer([first_valid_idx])[0]
                    ]
                    if isinstance(first_val, np.ndarray) or (
                        hasattr(first_val, "__len__") and not isinstance(first_val, str)
                    ):
                        obj_cols.append(col)

        if obj_cols:
            print(f"Removing {len(obj_cols)} array-type columns: {obj_cols}")
            df_causal = df_causal.drop(columns=obj_cols)

        # 2. Convert categorical columns to strings
        cat_cols = df_causal.select_dtypes(include=["category"]).columns.tolist()
        if cat_cols:
            print(
                f"Converting {len(cat_cols)} categorical columns to strings: {cat_cols}"
            )
            for col in cat_cols:
                df_causal[col] = df_causal[col].astype(str)

        # 3. Fill missing values with means for numeric columns and mode for categorical/string
        print("Filling missing values...")
        for col in df_causal.columns:
            # Skip the exposure and outcome for now
            if col not in [treatment, outcome]:
                if pd.api.types.is_numeric_dtype(df_causal[col]):
                    col_mean = df_causal[col].mean()
                    df_causal[col] = df_causal[col].fillna(col_mean)
                    print(
                        f"  - Filled {df_causal[col].isna().sum()} NaNs in '{col}' with mean {col_mean:.2f}"
                    )
                else:
                    if not df_causal[col].mode().empty:
                        col_mode = df_causal[col].mode().iloc[0]
                        df_causal[col] = df_causal[col].fillna(col_mode)
                        print(
                            f"  - Filled {df_causal[col].isna().sum()} NaNs in '{col}' with mode '{col_mode}'"
                        )
                    else:
                        df_causal[col] = df_causal[col].fillna("unknown")
                        print(
                            f"  - Filled {df_causal[col].isna().sum()} NaNs in '{col}' with 'unknown'"
                        )

        # 4. Ensure treatment and outcome exist and have no NaNs
        if treatment not in df_causal.columns:
            print(f"ERROR: Treatment '{treatment}' not found in DataFrame columns")

            # Log the error and use manual confounders if available
            if causal_log_file is not None and has_manual_confounders:
                with open(causal_log_file, "a") as f:
                    f.write(
                        f"ERROR: Treatment '{treatment}' not found in DataFrame columns\n"
                    )
                    f.write(f"Using manual confounders instead: {manual_conf_list}\n")

            return manual_conf_list if has_manual_confounders else ["age", "sex"]

        if outcome not in df_causal.columns:
            print(f"ERROR: Outcome '{outcome}' not found in DataFrame columns")

            # Log the error and use manual confounders if available
            if causal_log_file is not None and has_manual_confounders:
                with open(causal_log_file, "a") as f:
                    f.write(
                        f"ERROR: Outcome '{outcome}' not found in DataFrame columns\n"
                    )
                    f.write(f"Using manual confounders instead: {manual_conf_list}\n")

            return manual_conf_list if has_manual_confounders else ["age", "sex"]

        # Fill NaNs in treatment and outcome if any
        if df_causal[treatment].isna().any():
            if pd.api.types.is_numeric_dtype(df_causal[treatment]):
                df_causal[treatment] = df_causal[treatment].fillna(
                    df_causal[treatment].mean()
                )
            else:
                df_causal[treatment] = df_causal[treatment].fillna(
                    df_causal[treatment].mode().iloc[0]
                    if not df_causal[treatment].mode().empty
                    else "unknown"
                )
            print(f"WARNING: Filled NaNs in treatment variable '{treatment}'")

        if df_causal[outcome].isna().any():
            if pd.api.types.is_numeric_dtype(df_causal[outcome]):
                df_causal[outcome] = df_causal[outcome].fillna(
                    df_causal[outcome].mean()
                )
            else:
                df_causal[outcome] = df_causal[outcome].fillna(
                    df_causal[outcome].mode().iloc[0]
                    if not df_causal[outcome].mode().empty
                    else "unknown"
                )
            print(f"WARNING: Filled NaNs in outcome variable '{outcome}'")

        # Final check for remaining NaNs
        remaining_nas = df_causal.isna().sum().sum()
        if remaining_nas > 0:
            print(
                f"WARNING: {remaining_nas} NaN values remain in the DataFrame after preprocessing"
            )
            print(
                "Columns with NaNs:", df_causal.columns[df_causal.isna().any()].tolist()
            )
            # Drop rows with any remaining NaNs
            df_causal = df_causal.dropna()
            print(f"After dropping NaN rows, {len(df_causal)} rows remain")

        # Print some information about the graph
        print(f"Causal graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        print(f"Graph nodes: {list(graph.nodes)}")
        print(
            f"Creating CausalModel with {len(df_causal)} rows, treatment='{treatment}', outcome='{outcome}'"
        )

        # Create a causal model with DoWhy
        try:
            model = CausalModel(
                data=df_causal, treatment=treatment, outcome=outcome, graph=graph
            )

            # Identify estimand using backdoor criterion
            identified_estimand = model.identify_effect(
                proceed_when_unidentifiable=True
            )

            # Extract adjustment set
            try:
                # Try to access backdoor_variables directly
                backdoor_variables = None
                try:
                    # Since type checking doesn't like direct attribute access,
                    # we'll handle this at runtime without type checking
                    if hasattr(identified_estimand, "backdoor_variables"):
                        backdoor_variables = getattr(
                            identified_estimand, "backdoor_variables"
                        )
                    elif hasattr(identified_estimand, "get_backdoor_variables"):
                        backdoor_variables = getattr(
                            identified_estimand, "get_backdoor_variables"
                        )()
                except Exception as inner_e:
                    print(f"Error accessing backdoor_variables: {inner_e}")
                    backdoor_variables = None

                # Check if we were able to extract backdoor variables directly
                if backdoor_variables is not None:
                    # Filter to only variables in the DataFrame
                    confounders = [v for v in backdoor_variables if v in df.columns]
                    print(
                        f"DoWhy identified confounders for {target_relationship}: {confounders}"
                    )

                    # Update the causal log file with identified confounders
                    if causal_log_file is not None:
                        with open(causal_log_file, "a") as f:
                            f.write(f"AUTOMATED APPROACH (DoWhy):\n")
                            f.write(f"DoWhy identified confounders: {confounders}\n")

                            # Compare with manual approach
                            if has_manual_confounders:
                                f.write("\nCOMPARING APPROACHES:\n")
                                f.write(f"Manual confounders: {manual_conf_list}\n")
                                f.write(f"DoWhy confounders: {confounders}\n")

                                # Find agreement and differences
                                common = set(manual_conf_list).intersection(
                                    set(confounders)
                                )
                                manual_only = set(manual_conf_list) - set(confounders)
                                dowhy_only = set(confounders) - set(manual_conf_list)

                                f.write(f"Agreement (in both): {list(common)}\n")
                                f.write(f"Manual only: {list(manual_only)}\n")
                                f.write(f"DoWhy only: {list(dowhy_only)}\n\n")

                            # Decide which approach to use
                            if not confounders and has_manual_confounders:
                                f.write(
                                    "DECISION: Using manual confounders since DoWhy returned empty set.\n"
                                )
                                f.write(f"Final confounders: {manual_conf_list}\n")
                            else:
                                f.write(
                                    "DECISION: Using DoWhy identified confounders.\n"
                                )
                                f.write(f"Final confounders: {confounders}\n")

                    # Check if DoWhy returned an empty set - if so, use manual confounders
                    if not confounders and has_manual_confounders:
                        print(
                            f"DoWhy returned empty confounder set. Using manual confounders based on DAG: {manual_conf_list}"
                        )
                        return manual_conf_list

                    return confounders

                # If we get here, we need to try parsing from string representation
                estimand_str = str(identified_estimand)
                if "backdoor variables: " in estimand_str:
                    # Parse the backdoor variables from the string
                    backdoor_str = estimand_str.split("backdoor variables: ")[1].split(
                        "}"
                    )[0]
                    confounders = [
                        v.strip()
                        for v in backdoor_str.split(",")
                        if v.strip() in df.columns
                    ]
                    print(f"Extracted confounders from estimand string: {confounders}")

                    # Update the causal log file with identified confounders
                    if causal_log_file is not None:
                        with open(causal_log_file, "a") as f:
                            f.write(f"AUTOMATED APPROACH (string parsing):\n")
                            f.write(f"DoWhy identified confounders: {confounders}\n")

                            # Compare with manual approach
                            if has_manual_confounders:
                                f.write("\nCOMPARING APPROACHES:\n")
                                f.write(f"Manual confounders: {manual_conf_list}\n")
                                f.write(f"DoWhy confounders: {confounders}\n")

                                # Find agreement and differences
                                common = set(manual_conf_list).intersection(
                                    set(confounders)
                                )
                                manual_only = set(manual_conf_list) - set(confounders)
                                dowhy_only = set(confounders) - set(manual_conf_list)

                                f.write(f"Agreement (in both): {list(common)}\n")
                                f.write(f"Manual only: {list(manual_only)}\n")
                                f.write(f"DoWhy only: {list(dowhy_only)}\n\n")

                            # Decide which approach to use
                            if not confounders and has_manual_confounders:
                                f.write(
                                    "DECISION: Using manual confounders since DoWhy returned empty set.\n"
                                )
                                f.write(f"Final confounders: {manual_conf_list}\n")
                            else:
                                f.write(
                                    "DECISION: Using DoWhy identified confounders.\n"
                                )
                                f.write(f"Final confounders: {confounders}\n")

                    # Check if this method also returned an empty set
                    if not confounders and has_manual_confounders:
                        print(
                            f"Estimand string parsing returned empty confounder set. Using manual confounders based on DAG: {manual_conf_list}"
                        )
                        return manual_conf_list

                    return confounders
                else:
                    print(
                        f"No backdoor adjustment needed for {target_relationship} according to DoWhy"
                    )

                    # Update the causal log file
                    if causal_log_file is not None:
                        with open(causal_log_file, "a") as f:
                            f.write("AUTOMATED APPROACH:\n")
                            f.write("DoWhy suggests no backdoor adjustment needed.\n")

                            # Compare with manual approach
                            if has_manual_confounders:
                                f.write("\nCOMPARING APPROACHES:\n")
                                f.write(f"Manual confounders: {manual_conf_list}\n")
                                f.write(
                                    "DoWhy confounders: None (no adjustment needed)\n\n"
                                )

                                # Decide which approach to use
                                f.write(
                                    "DECISION: Using manual confounders despite DoWhy suggesting no adjustment.\n"
                                )
                                f.write(f"Final confounders: {manual_conf_list}\n")
                            else:
                                f.write(
                                    "DECISION: No confounders needed for adjustment.\n"
                                )
                                f.write("Final confounders: []\n")

                    # Even if DoWhy says no adjustment needed, we may want to use our manual confounders
                    if has_manual_confounders:
                        print(
                            f"DoWhy suggests no adjustment, but using manual confounders based on DAG: {manual_conf_list}"
                        )
                        return manual_conf_list

                    return []
            except Exception as e:
                print(f"Error extracting confounders: {e}")
                # Default confounders based on the relationship
                print(
                    f"WARNING: Falling back to default confounders for {target_relationship}"
                )

                # Log the error
                if causal_log_file is not None:
                    with open(causal_log_file, "a") as f:
                        f.write(f"ERROR in automated approach: {e}\n")
                        if has_manual_confounders:
                            f.write(
                                f"Falling back to manual confounders: {manual_conf_list}\n"
                            )
                            f.write(f"Final confounders: {manual_conf_list}\n")
                        else:
                            f.write(
                                "No manual confounders available, using defaults.\n"
                            )

                if has_manual_confounders:
                    return manual_conf_list
                elif "patient_chronotype" in treatment:
                    fallback = ["age", "sex", "sri"]
                    if causal_log_file is not None:
                        with open(causal_log_file, "a") as f:
                            f.write(
                                f"Using default chronotype confounders: {fallback}\n"
                            )
                    return fallback
                elif "nadir_time_pct" in treatment:
                    fallback = ["age", "sex", "patient_chronotype", "sri"]
                    if causal_log_file is not None:
                        with open(causal_log_file, "a") as f:
                            f.write(
                                f"Using default nadir_time confounders: {fallback}\n"
                            )
                    return fallback
                else:
                    fallback = ["age", "sex"]
                    if causal_log_file is not None:
                        with open(causal_log_file, "a") as f:
                            f.write(f"Using minimal default confounders: {fallback}\n")
                    return fallback

        except Exception as dowhy_e:
            print(f"Error in DoWhy CausalModel creation: {dowhy_e}")
            # Fallback based on target relationship
            print(
                f"WARNING: Falling back to default confounders for {target_relationship}"
            )

            # Log the error
            if causal_log_file is not None:
                with open(causal_log_file, "a") as f:
                    f.write(f"ERROR in DoWhy model creation: {dowhy_e}\n")
                    if has_manual_confounders:
                        f.write(
                            f"Falling back to manual confounders: {manual_conf_list}\n"
                        )

            if has_manual_confounders:
                if causal_log_file is not None:
                    with open(causal_log_file, "a") as f:
                        f.write(f"Final confounders: {manual_conf_list}\n")
                return manual_conf_list
            elif "patient_chronotype" in treatment:
                fallback = ["age", "sex", "sri"]
                if causal_log_file is not None:
                    with open(causal_log_file, "a") as f:
                        f.write(f"Final confounders: {fallback}\n")
                return fallback
            elif "nadir_time_pct" in treatment:
                fallback = ["age", "sex", "patient_chronotype", "sri"]
                if causal_log_file is not None:
                    with open(causal_log_file, "a") as f:
                        f.write(f"Final confounders: {fallback}\n")
                return fallback
            else:
                fallback = ["age", "sex"]
                if causal_log_file is not None:
                    with open(causal_log_file, "a") as f:
                        f.write(f"Final confounders: {fallback}\n")
                return fallback

    except Exception as e:
        print(f"Error in causal inference process: {e}")

        # More verbose error logging with traceback
        import traceback

        print("--- Detailed traceback ---")
        traceback.print_exc()
        print("-------------------------")

        # Update the log file with the error
        if causal_log_file is not None:
            with open(causal_log_file, "a") as f:
                f.write(f"ERROR in causal inference process: {e}\n")
                f.write("--- Detailed traceback ---\n")
                f.write(traceback.format_exc())
                f.write("-------------------------\n")

                if has_manual_confounders:
                    f.write(f"Falling back to manual confounders: {manual_conf_list}\n")
                    f.write(f"Final confounders: {manual_conf_list}\n")

        # Fallback to sensible defaults based on domain knowledge
        print(f"WARNING: Falling back to default confounders for {target_relationship}")
        if has_manual_confounders:
            return manual_conf_list
        elif "patient_chronotype -> cluster" in target_relationship:
            fallback = ["age", "sex", "sri"]
            if causal_log_file is not None:
                with open(causal_log_file, "a") as f:
                    f.write(f"Final fallback to default confounders: {fallback}\n")
            return fallback
        elif "nadir_time_pct -> cluster" in target_relationship:
            fallback = ["age", "sex", "patient_chronotype", "sri"]
            if causal_log_file is not None:
                with open(causal_log_file, "a") as f:
                    f.write(f"Final fallback to default confounders: {fallback}\n")
            return fallback
        elif "se -> cluster" in target_relationship:
            fallback = ["age", "sex", "sfi", "waso"]
            if causal_log_file is not None:
                with open(causal_log_file, "a") as f:
                    f.write(f"Final fallback to default confounders: {fallback}\n")
            return fallback
        else:
            fallback = ["age", "sex"]
            if causal_log_file is not None:
                with open(causal_log_file, "a") as f:
                    f.write(f"Final fallback to default confounders: {fallback}\n")
            return fallback


def render_save_causal_dag(
    output_dir: Union[str, Path],
    filename: str = "sleep_causal_dag",
    format: str = "png",
    view: bool = False,
) -> Optional[Path]:
    """
    Renders and saves the sleep causal DAG as an image.

    Parameters
    ----------
    output_dir : str or Path
        Directory where the image will be saved
    filename : str, default="sleep_causal_dag"
        Base filename for the image (without extension)
    format : str, default="png"
        Image format (png, pdf, svg, etc.)
    view : bool, default=False
        Whether to open the image after saving

    Returns
    -------
    Path or None
        Path to the saved image, or None if rendering failed
    """
    # Check if graphviz is available
    if not GRAPHVIZ_AVAILABLE:
        print("Error: graphviz package is not installed. Run 'pip install graphviz'.")
        print("You also need the Graphviz software installed on your system.")
        return None

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get the DAG as a NetworkX graph
    G = create_sleep_causal_graph()

    try:
        # Try different approaches to visualize the graph
        try:
            # First approach: Use NetworkX's agraph interface (requires pygraphviz)
            from networkx.drawing.nx_agraph import to_agraph

            A = to_agraph(G)

            # Add formatting
            A.graph_attr.update(fontsize="10", rankdir="LR")
            A.node_attr.update(shape="ellipse", style="filled", fillcolor="lightblue")

            # Save to file
            dot_file = output_path / f"{filename}.dot"
            A.write(str(dot_file))

            # Use graphviz to render the image
            if graphviz is not None:
                graph = graphviz.Source.from_file(str(dot_file))
                rendered_file = graph.render(
                    filename=filename,
                    directory=str(output_path),
                    format=format,
                    cleanup=False,
                )

                print(
                    f"Successfully rendered DAG image using pygraphviz: {rendered_file}"
                )
                return Path(rendered_file)
            else:
                print("Graphviz module not available to render from dot file")
                raise ImportError("Graphviz module not available")

        except ImportError as e:
            print(f"Could not use pygraphviz: {e}")
            print("Trying alternative method with plain graphviz...")

            # Second approach: Use graphviz directly if available
            if graphviz is not None:
                dot = graphviz.Digraph(comment="Sleep Causal DAG")
                dot.attr(rankdir="LR")

                # Add all nodes
                for node in G.nodes():
                    dot.node(
                        node, shape="ellipse", style="filled", fillcolor="lightblue"
                    )

                # Add all edges
                for edge in G.edges():
                    dot.edge(edge[0], edge[1])

                # Render the graph
                output_file = dot.render(
                    filename=filename,
                    directory=str(output_path),
                    format=format,
                    cleanup=False,
                )

                print(
                    f"Successfully rendered DAG image using direct graphviz: {output_file}"
                )
                return Path(output_file)
            else:
                print("Graphviz module not available for direct graph creation")
                raise ImportError("Graphviz module not available")

    except Exception as e:
        print(f"Error rendering causal DAG: {e}")
        print("Please ensure Graphviz is installed and in your system's PATH.")

        # Fallback to matplotlib for basic visualization if available
        if MATPLOTLIB_AVAILABLE and plt is not None:
            try:
                plt.figure(figsize=(12, 8))
                pos = nx.spring_layout(G)
                nx.draw(
                    G,
                    pos,
                    with_labels=True,
                    node_color="lightblue",
                    node_size=2000,
                    edge_color="gray",
                    arrows=True,
                    font_size=10,
                    font_weight="bold",
                )

                # Save the figure
                img_path = output_path / f"{filename}.{format}"
                plt.savefig(img_path, format=format, bbox_inches="tight")
                plt.close()

                print(f"Rendered DAG using matplotlib as fallback: {img_path}")
                return img_path
            except Exception as plt_err:
                print(f"Failed to use matplotlib for visualization: {plt_err}")
                return None
        else:
            print("Neither Graphviz nor Matplotlib are available for visualization")
            return None
