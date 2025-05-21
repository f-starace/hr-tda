"""
Features module for sleep data analysis.

This module contains functions for feature engineering,
such as equalizing HR time series length, calculating slopes,
and computing topological data analysis (TDA) features.
"""

from typing import Tuple

import numpy as np
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy
from tslearn.preprocessing import TimeSeriesResampler
from tslearn.piecewise import PiecewiseAggregateApproximation

# Set constants for feature generation
TARGET_LEN = 100  # Target length for equalized heart rate time series
WINDOW_SIZE = 30  # Window size for time-delay embedding


def equalize_hr(
    hr: np.ndarray, method: str = "resample", target_len: int = TARGET_LEN
) -> np.ndarray:
    """Return equally sized HR via interpolation or PAA.

    Parameters
    ----------
    hr : np.ndarray
        Raw heart-rate series (1-D).
    method : {"resample", "paa"}
        * "resample" → cubic spline / linear interpolation to `target_len`.
        * "paa"      → Piecewise Aggregate Approximation with `target_len` segments.
    target_len : int
        Desired length.
    """
    # Make sure data is compatible with tslearn's expected format
    # tslearn expects (n_samples, n_timestamps, n_features) or (n_timestamps,)
    # Convert to a flat 1D array first
    hr_flat = np.asarray(hr).flatten()

    # Check if we have enough data
    if len(hr_flat) < 2:
        print(
            f"Warning: Input HR has only {len(hr_flat)} points, which is not enough for processing."
        )
        return np.zeros(target_len, dtype=np.float32)

    if method == "resample":
        # For TimeSeriesResampler, reshape to (1, n_timestamps)
        hr_reshaped = hr_flat.reshape(1, -1)
        eq = TimeSeriesResampler(sz=target_len).fit_transform(hr_reshaped)[0]
    elif method == "paa":
        # For PAA, the data needs to be 2D: (n_samples, n_timestamps)
        # PAA requires input time series length to be greater than or equal to n_segments (target_len)
        if len(hr_flat) < target_len:
            print(
                f"Warning: HR length {len(hr_flat)} for PAA is less than target_len {target_len}. Resampling to {target_len} first."
            )
            hr_reshaped = hr_flat.reshape(1, -1)
            hr_resampled_for_paa = TimeSeriesResampler(sz=target_len).fit_transform(
                hr_reshaped
            )[0]
            hr_flat = hr_resampled_for_paa.flatten()  # Now hr_flat has target_len

        # Apply PAA with the properly sized input
        paa = PiecewiseAggregateApproximation(n_segments=target_len)
        hr_reshaped = hr_flat.reshape(1, -1)
        eq = paa.fit_transform(hr_reshaped)[0]
    else:
        raise ValueError(f"Unknown equalize_method: {method}")

    # Ensure we return a flat 1D array
    return eq.flatten().astype(np.float32)


def slope_fixed_len(hr: np.ndarray, method: str = "resample") -> np.ndarray:
    """Equalise then return HR slope (first derivative)."""
    hr_eq_potentially_2d = equalize_hr(hr, method=method)

    # Ensure hr_eq is 1D before further processing and gradient calculation
    hr_eq_1d = hr_eq_potentially_2d.squeeze()

    # Safeguard against the 1D hr_eq_1d being too short for np.gradient
    if len(hr_eq_1d) < 2:
        print(
            f"Warning: Squeezed hr_eq_1d length is {len(hr_eq_1d)} (from original shape {hr_eq_potentially_2d.shape}) "
            f"before gradient calculation, which is less than the required 2 elements. "
            f"Expected target length {TARGET_LEN}. Input hr length was {len(hr)}. Method: {method}. "
            f"Returning zeros of length {TARGET_LEN}."
        )
        return np.zeros(TARGET_LEN, dtype=np.float32)

    slope = np.gradient(hr_eq_1d)
    return slope.astype(np.float32)


def create_time_delay_embedding(time_series, dim=3, tau=1):
    """
    Create a time-delay embedding from a 1D time series.

    Parameters
    ----------
    time_series : np.ndarray
        1D time series data
    dim : int, default=3
        Embedding dimension
    tau : int, default=1
        Time delay

    Returns
    -------
    np.ndarray
        Time-delay embedding with shape (n_points, dim)
    """
    n = len(time_series)
    points = n - (dim - 1) * tau

    if points <= 0:
        return np.array([])

    embedding = np.zeros((points, dim))

    for i in range(points):
        for j in range(dim):
            embedding[i, j] = time_series[i + j * tau]

    return embedding


# Define TDA components for direct use
vietoris_rips = VietorisRipsPersistence(metric="euclidean", homology_dimensions=(0, 1))
persistence_entropy = PersistenceEntropy()


def tda_vector(hr_slope: np.ndarray) -> np.ndarray:
    """Compute a persistence-entropy vector from the slope series."""
    # First check if the data is long enough
    min_len = WINDOW_SIZE * 2  # Need enough points for meaningful embedding
    if len(hr_slope) < min_len:
        print(
            f"Warning: Input to tda_vector has length {len(hr_slope)}, "
            f"which is less than minimum length {min_len}. "
            f"Returning zeros array of length 2."
        )
        return np.zeros(2, dtype=np.float32)

    try:
        # 1. Create time delay embedding manually
        embedding = create_time_delay_embedding(
            hr_slope,
            dim=3,  # Embedding dimension
            tau=5,  # Time delay (adjust for your data)
        )

        if len(embedding) == 0:
            return np.zeros(2, dtype=np.float32)

        # 2. Apply Vietoris-Rips directly to the embedding
        # Convert to list of points for VietorisRips
        diagrams = vietoris_rips.fit_transform(np.array([embedding]))

        # 3. Extract persistence entropy features
        entropy_features = persistence_entropy.fit_transform(diagrams)

        return entropy_features[0].astype(np.float32)
    except Exception as e:
        print(
            f"Error in TDA processing: {e}. Input shape: {hr_slope.shape}. Returning zeros."
        )
        return np.zeros(2, dtype=np.float32)
