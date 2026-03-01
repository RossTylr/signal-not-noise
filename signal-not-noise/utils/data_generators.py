"""
Synthetic data generators for signal-not-noise.
Consistent datasets used across notebooks and the Streamlit app.
"""

import numpy as np
from sklearn.datasets import make_swiss_roll, load_digits


def make_patient_data(n=200, d_signal=3, d_noise=0, seed=42):
    """
    Generate synthetic patient data with controlled signal and noise dimensions.

    Signal dimensions: age, heart_rate, blood_oxygen (with realistic correlations).
    Noise dimensions: random uniform features with no relationship to signal.

    Returns:
        X: (n, d_signal + d_noise) array
        feature_names: list of strings
        signal_mask: boolean array (True for signal features)
    """
    rng = np.random.default_rng(seed)

    age = rng.normal(45, 15, n).clip(18, 90)
    heart_rate = 0.3 * age + rng.normal(70, 10, n)
    blood_ox = rng.normal(97, 2, n).clip(80, 100)

    signal_features = [age, heart_rate, blood_ox]
    signal_names = ["Age", "Heart Rate", "SpO2"]

    if d_signal > 3:
        # Add more correlated signal features
        bp_systolic = 0.5 * age + rng.normal(120, 15, n)
        signal_features.append(bp_systolic)
        signal_names.append("BP Systolic")

    if d_signal > 4:
        resp_rate = -0.1 * blood_ox + rng.normal(16, 3, n)
        signal_features.append(resp_rate)
        signal_names.append("Resp Rate")

    # Use only requested number of signal dims
    signal_features = signal_features[:d_signal]
    signal_names = signal_names[:d_signal]

    X_signal = np.column_stack(signal_features)

    # Add noise dimensions
    noise_names = [f"Noise_{i}" for i in range(d_noise)]
    if d_noise > 0:
        X_noise = rng.uniform(0, 100, size=(n, d_noise))
        X = np.hstack([X_signal, X_noise])
    else:
        X = X_signal

    feature_names = signal_names + noise_names
    signal_mask = np.array([True] * len(signal_names) +
                           [False] * len(noise_names))

    return X, feature_names, signal_mask


def make_low_d_in_high_d(n=500, intrinsic_d=3, ambient_d=100, seed=42):
    """
    Generate data that lives on a low-dimensional manifold
    embedded in a high-dimensional space.

    The data is generated in intrinsic_d dimensions, then projected
    into ambient_d dimensions via a random linear map plus small noise.
    """
    rng = np.random.default_rng(seed)

    # Generate low-dimensional data
    Z = rng.standard_normal((n, intrinsic_d))

    # Random projection matrix
    W = rng.standard_normal((intrinsic_d, ambient_d))

    # Project and add small noise
    X = Z @ W + rng.standard_normal((n, ambient_d)) * 0.1

    return X, Z


def make_swiss_roll_data(n=1000, noise=0.5, seed=42):
    """Swiss roll: classic nonlinear manifold example."""
    X, colour = make_swiss_roll(n_samples=n, noise=noise, random_state=seed)
    return X, colour


def make_two_class_with_noise(n=300, d_noise_list=None, seed=42):
    """
    A 2D classification problem embedded in increasing noise dimensions.
    Used to demonstrate how noise features degrade classifier performance.

    Returns dict mapping total_dims -> (X, y)
    """
    if d_noise_list is None:
        d_noise_list = [0, 5, 10, 20, 50, 100, 200, 500]

    rng = np.random.default_rng(seed)
    X_signal = rng.standard_normal((n, 2))
    y = (X_signal[:, 0] + X_signal[:, 1] > 0).astype(int)

    datasets = {}
    for d_noise in d_noise_list:
        if d_noise > 0:
            X_noise = rng.standard_normal((n, d_noise))
            X = np.hstack([X_signal, X_noise])
        else:
            X = X_signal.copy()
        datasets[2 + d_noise] = (X, y)

    return datasets


def make_digit_data():
    """Load sklearn digits dataset (8x8 images = 64D points)."""
    digits = load_digits()
    return digits.data, digits.target, digits.images
