"""
Preprocessing module for sleep data analysis.

This module contains functions for cleaning and preprocessing heart rate data
and unpacking sleep episodes from raw data sources.
"""

from __future__ import annotations

import ast
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def clean_hr(hr: np.ndarray, bpm_low: int = 25, bpm_high: int = 220) -> np.ndarray:
    """
    Clean and preprocess heart rate time series.

    Parameters
    ----------
    hr : np.ndarray
        Raw heart rate time series
    bpm_low : int, default=25
        Lower bound for valid heart rate values
    bpm_high : int, default=220
        Upper bound for valid heart rate values

    Returns
    -------
    np.ndarray
        Cleaned heart rate series with interpolated middle values and edge NaNs removed
    """
    hr = hr.astype(float)

    # Create a mask for extreme values
    extreme_mask = (hr < bpm_low) | (hr > bpm_high)

    # Set extreme values to NaN
    hr_clean = hr.copy()
    hr_clean[extreme_mask] = np.nan

    # Find contiguous blocks of valid values
    valid_indices = np.where(~np.isnan(hr_clean))[0]

    if len(valid_indices) == 0:
        # No valid data points, return empty array
        return np.array([], dtype=float)

    # Get start and end of valid range
    start_idx = valid_indices[0]
    end_idx = valid_indices[-1]

    # Extract the portion of the array with valid values at the ends
    hr_subset = hr_clean[start_idx : end_idx + 1]

    # Interpolate internal NaN values
    mask_subset = np.isnan(hr_subset)
    if np.any(mask_subset) and len(hr_subset) > 1:
        # Create indices for interpolation
        x = np.arange(len(hr_subset))
        # Get positions of valid values
        x_valid = x[~mask_subset]
        # Get valid values
        y_valid = hr_subset[~mask_subset]

        # Only interpolate if we have at least two valid points
        if len(x_valid) >= 2:
            # Interpolate missing values
            hr_subset[mask_subset] = np.interp(x[mask_subset], x_valid, y_valid)

    return hr_subset


def extract_hr_nadir_features(
    hr: np.ndarray,
    sleep_duration_secs: float,
    hr_times: List[float],
    hr_values: List[float],
) -> Dict[str, float]:
    """
    Extract features related to the timing of the lowest heart rate (nadir).

    Parameters
    ----------
    hr : np.ndarray
        Clean heart rate time series (not used if hr_times and hr_values are provided)
    sleep_duration_secs : float
        Total sleep duration in seconds
    hr_times : List[float]
        Timestamps for each HR measurement as seconds from sleep onset
    hr_values : List[float]
        Heart rate values corresponding to hr_times

    Returns
    -------
    Dict[str, float]
        Dictionary with nadir timing features
    """
    # Convert to numpy arrays for easier handling
    times_array = np.array(hr_times)
    values_array = np.array(hr_values)

    # Find the index of the minimum heart rate
    nadir_idx = np.argmin(values_array)

    # Get the nadir time in seconds from sleep onset
    nadir_time_secs = times_array[nadir_idx]

    # Convert to hours
    nadir_time_hours = nadir_time_secs / 3600.0

    # Calculate as percentage of sleep period
    nadir_time_pct = (
        nadir_time_secs / sleep_duration_secs if sleep_duration_secs > 0 else 0.5
    )

    # Get the minimum heart rate value
    nadir_hr = values_array[nadir_idx]

    # Calculate normalized time to nadir (for comparison with Fudolig et al.)
    time_to_nadir_normalized = nadir_time_pct

    # Ensure all values are native Python types to avoid type issues
    return {
        "nadir_time_hours": float(nadir_time_hours),
        "nadir_time_pct": float(nadir_time_pct),
        "nadir_hr": float(nadir_hr),
        "time_to_nadir_normalized": float(time_to_nadir_normalized),
    }


def episode_iter(row: pd.Series) -> List[dict]:
    """Yield each episode dict from the JSON-like cell."""
    episodes = row.get("sleep_episodes")

    # Check if episodes is iterable and contains dictionaries
    if not isinstance(episodes, (list, tuple)):
        return []

    # Ensure all elements are dictionaries and convert to list explicitly
    result = []
    for ep in episodes:
        if isinstance(ep, dict):
            result.append(ep)

    return result


def unpack_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten patient-level DataFrame into one row per sleep episode.
    Also computes a **continuous chronotype** as hours since 18:00 to MSP.
    """
    records = []
    for idx, row in df.iterrows():
        # Convert idx to integer for comparisons if needed
        idx_num = int(idx) if isinstance(idx, (int, np.integer)) else 0

        episodes = episode_iter(row)
        if not episodes and idx_num < 5:
            print(f"Warning: No episodes found for row {idx}")

        for ep_idx, ep in enumerate(episodes):

            try:
                msp = pd.to_datetime(ep["msp"])
                # hours difference from 18:00, wrap into [0,24)
                total_min = msp.hour * 60 + msp.minute
                diff = total_min - (18 * 60)
                if diff < 0:
                    diff += 24 * 60
                chron_cont = diff / 60.0

                # Use get() with default values for potentially missing keys
                rec = {
                    "id": row["id"],
                    "episode_id": f"{row['id']}__{ep.get('onset', 'unknown')}",
                    "duration_secs": ep.get("sleep_episode_duration_secs", 0),
                    "waso": ep.get("waso_min", 0),
                    "se": ep.get("se_percent", 0),
                    "sfi": ep.get("sleep_fragmentation_index", 0),
                    "activity_idx": ep.get("activity_index", 0),
                    "msp": msp,
                    "msp_time": msp.time(),  # Store the time component for desync calculation
                    "chronotype_cont": chron_cont,
                    "is_main": ep.get("is_main", False),
                }

                # Handle heart rate data with care
                if "hr_vector_values" in ep:
                    rec["hr_vector"] = ep["hr_vector_values"]
                    if idx_num < 3 and ep_idx == 0:
                        print(
                            f"Using hr_vector_values, type: {type(ep['hr_vector_values'])}, length: {len(ep['hr_vector_values']) if hasattr(ep['hr_vector_values'], '__len__') else 'N/A'}"
                        )
                elif "hr_vector" in ep:
                    # If hr_vector_values is missing but hr_vector exists, try that
                    if idx_num < 3 and ep_idx == 0:
                        print(
                            f"Using hr_vector as fallback, type: {type(ep['hr_vector'])}"
                        )
                    rec["hr_vector"] = ep.get("hr_vector", [])
                else:
                    rec["hr_vector"] = []
                    if idx_num < 3 and ep_idx == 0:
                        print("No heart rate data found in episode")

                # Store original hr_vector field as hr_vector_times
                rec["hr_vector_times"] = ep.get("hr_vector", [])

                # Extract hr_times and hr_values if available
                if "hr_times" in ep:
                    rec["hr_times"] = ep["hr_times"]
                if "hr_values" in ep:
                    rec["hr_values"] = ep["hr_values"]

                # Alternative field names
                if "hr_time_seconds" in ep:
                    rec["hr_times"] = ep["hr_time_seconds"]
                if "hr_values_raw" in ep:
                    rec["hr_values"] = ep["hr_values_raw"]

                # copy other patient-level covariates
                for col in ["age", "sex", "resting_hr", "sri"]:
                    if col in row:
                        rec[col] = row[col]

                records.append(rec)
            except Exception as e:
                if idx_num < 5:
                    print(f"Error processing row {idx}, episode {ep_idx}: {str(e)}")
                continue

    result_df = pd.DataFrame(records)

    # Print summary stats
    print(f"Processed {len(df)} input rows into {len(result_df)} episode records")
    print(
        f"hr_vector has data: {sum(len(x) > 0 if hasattr(x, '__len__') else 0 for x in result_df['hr_vector'])}/{len(result_df)} episodes"
    )

    return result_df


def calculate_patient_chronotype(episodes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average chronotype per patient and add it to the DataFrame.
    Also computes chronotype desynchronization.

    Parameters
    ----------
    episodes_df : pd.DataFrame
        DataFrame containing sleep episodes with chronotype_cont column

    Returns
    -------
    pd.DataFrame
        The input DataFrame with new patient_chronotype and chronotype_desync columns
    """
    # Make a copy to avoid SettingWithCopyWarning
    df_result = episodes_df.copy()

    # Calculate average chronotype per patient
    patient_chronotypes = df_result.groupby("id")["chronotype_cont"].mean()

    # Map the average chronotype back to each episode
    df_result["patient_chronotype"] = df_result["id"].map(patient_chronotypes)

    # Calculate chronotype desynchronization (difference from average)
    df_result["chronotype_desync"] = (
        df_result["chronotype_cont"] - df_result["patient_chronotype"]
    )

    # Get absolute desynchronization
    df_result["abs_chronotype_desync"] = abs(df_result["chronotype_desync"])

    print(
        f"Calculated patient chronotypes: range {patient_chronotypes.min():.2f} to "
        f"{patient_chronotypes.max():.2f} hours since 18:00"
    )
    print(
        f"Chronotype desynchronization range: {df_result['chronotype_desync'].min():.2f} to "
        f"{df_result['chronotype_desync'].max():.2f} hours"
    )

    return df_result
