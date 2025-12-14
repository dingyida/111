#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute "style error" for multi-view reconstruction coverage.

We implement:
    Q = 1 - exp( -k * mean_{ω in S^2}[ C(ω) ] )
with
    C(ω) = 1 - Π_i (1 - a_i(ω))
    a_i(ω) = w_i * exp( - (γ_i(ω) / φ_i)^p )
and merged per-camera weight
    w_i = ( r_i^β ) / (1 + α d_i^2)^ζ .

Here:
- γ_i(ω) is the angular separation between a sample direction ω and camera i's
  viewing direction ω_i (unit vector from camera position to object position).
- φ_i is the camera's half-FOV (in radians). If per-row "fov_deg" exists we use it,
  otherwise a global --fov-deg is used.
- d_i is camera–object distance (meters). If per-row "distance" exists we use it,
  otherwise it’s computed from camera/object positions.
- r_i captures resolution; if columns "width" and "height" exist, pixels = width*height.
  If "pixels" exists we use it directly. We normalize r_i by pixels per steradian,
  using camera FOV if available; otherwise normalize by the median pixels value.
- style_error = 1 - Q.

USAGE EXAMPLES:
    python style_error.py --obs observations.csv --k 4 --p 8 --alpha 0.2 --beta 0.6 --zeta 1.0 --fov-deg 45 --samples 4096
    python style_error.py --obs observations.csv --k 3.5 --p 10 --alpha 0.25 --beta 0.7 --zeta 1 --fov-deg 50 --samples 8192 --save-coverage coverage.csv

CSV EXPECTED COLUMNS (robust parsing):
    Required (positions):
        camera_x, camera_y, camera_z,
        object_x, object_y, object_z
    Optional (if present, used):
        distance               # overrides computed distance
        fov_deg                # per-row half-FOV (degrees). If absent, use --fov-deg
        width, height          # image size (pixels = width*height)
        pixels                 # alternative to width*height
    Ignored columns are allowed; rotations are not required since ω_i uses positions.

OUTPUT:
    Prints JSON to stdout: {"Q": float, "style_error": float, "mean_coverage": float, ...}
    Optionally writes per-sample coverage C(ω) if --save-coverage is given.
"""
import argparse
import json
import math
from typing import Tuple, Optional

import numpy as np
import pandas as pd


def fibonacci_sphere(n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Uniform points on S^2 using a Fibonacci lattice; returns (n,3) unit vectors."""
    if n <= 0:
        raise ValueError("Number of samples must be positive")
    if rng is None:
        rng = np.random.default_rng(12345)
    ga = math.pi * (3 - math.sqrt(5.0))  # golden angle
    k = np.arange(n)
    z = (2 * k + 1) / n - 1
    r = np.sqrt(1 - z * z)
    theta = k * ga
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pts = np.stack([x, y, z], axis=1).astype(np.float64)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True) + 1e-12
    return pts


def solid_angle_from_fov_deg(fov_deg: float) -> float:
    """Solid angle of a spherical cap for half-FOV = fov_deg (degrees)."""
    phi = math.radians(fov_deg)
    return 2 * math.pi * (1 - math.cos(phi))


def parse_resolution_terms(df: pd.DataFrame, default_fov_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build per-row resolution scaler r_i and per-row φ_i (radians).
    - If width & height exist: pixels_i = width*height
    - Else if pixels exists: pixels_i = pixels
    - Else: pixels_i = 1.0
    Normalize r_i as pixels per steradian (pps) relative to median pps.
    Returns: (r_i, phi_i_rad, pixels)
    """
    n = len(df)
    if "fov_deg" in df.columns:
        fov_deg = df["fov_deg"].fillna(default_fov_deg).to_numpy(dtype=float)
    else:
        fov_deg = np.full(n, float(default_fov_deg), dtype=float)

    if "width" in df.columns and "height" in df.columns:
        pixels = (df["width"].astype(float) * df["height"].astype(float)).to_numpy()
    elif "pixels" in df.columns:
        pixels = df["pixels"].astype(float).to_numpy()
    else:
        pixels = np.ones(n, dtype=float)

    omega = np.array([solid_angle_from_fov_deg(x) for x in fov_deg])  # steradians
    omega = np.clip(omega, 1e-8, None)

    pps = pixels / omega  # pixels per steradian
    median_den = np.median(pps) if np.all(np.isfinite(pps)) and np.any(pps > 0) else 1.0
    r_i = pps / (median_den if median_den > 0 else 1.0)

    phi_i = np.radians(fov_deg)  # half-FOV in radians
    return r_i.astype(float), phi_i.astype(float), pixels.astype(float)


def compute_view_dirs_and_dist(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-row unit viewing directions ω_i and distances d_i:
      ω_i = normalize(object_pos - camera_pos)
      d_i = ||object_pos - camera_pos||  (overridden by 'distance' column if present)
    Returns (omega_i (N,3), d_i (N,))
    """
    cam = df[["camera_x", "camera_y", "camera_z"]].to_numpy(dtype=float)
    obj = df[["object_x", "object_y", "object_z"]].to_numpy(dtype=float)
    vec = obj - cam
    d = np.linalg.norm(vec, axis=1)
    omega = vec / (d[:, None] + 1e-12)

    if "distance" in df.columns:
        provided = df["distance"].astype(float).to_numpy()
        mask = np.isfinite(provided) & (provided > 0)
        d[mask] = provided[mask]
    return omega, d


def style_error_from_csv(
    csv_path: str,
    k: float,
    p: float,
    alpha: float,
    beta: float,
    zeta: float,
    default_fov_deg: float,
    n_samples: int,
    seed: int = 12345,
    save_coverage: Optional[str] = None,
) -> dict:
    df = pd.read_csv(csv_path)
    required = ["camera_x", "camera_y", "camera_z", "object_x", "object_y", "object_z"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    omegas_i, d_i = compute_view_dirs_and_dist(df)
    r_i, phi_i, _pixels = parse_resolution_terms(df, default_fov_deg)

    # Per-camera weight
    w_i = (r_i ** beta) / ((1.0 + alpha * (d_i ** 2)) ** zeta)
    w_i = np.clip(w_i, 0.0, 1.0)

    # Sphere sampling
    rng = np.random.default_rng(seed)
    samples = fibonacci_sphere(n_samples, rng)  # (M,3)

    # Compute a_i(ω) for all samples and cameras
    dot = np.clip(samples @ omegas_i.T, -1.0, 1.0)  # (M,N)
    gamma = np.arccos(dot)                          # (M,N)
    phi_safe = np.clip(phi_i, 1e-6, None)[None, :]
    a = w_i[None, :] * np.exp(- (gamma / phi_safe) ** p)
    a = np.clip(a, 0.0, 1.0)

    # Soft union over cameras
    C = 1.0 - np.prod(1.0 - a, axis=1)  # (M,)
    I = float(np.mean(C))               # coverage integral ≈ average

    Q = 1.0 - math.exp(-k * I)
    style_error = 1.0 - Q

    result = {
        "Q": Q,
        "style_error": style_error,
        "mean_coverage": I,
        "num_views": int(len(df)),
        "num_samples": int(n_samples),
        "k": k, "p": p, "alpha": alpha, "beta": beta, "zeta": zeta,
        "default_fov_deg": default_fov_deg,
    }

    if save_coverage:
        pd.DataFrame({"C_sample": C}).to_csv(save_coverage, index=False)

    return result


def main():
    ap = argparse.ArgumentParser(description="Compute style_error from observation CSV using coverage-based Q model.")
    ap.add_argument("--obs", required=True, help="Path to observation CSV.")
    ap.add_argument("--k", type=float, default=4.0, help="Global saturation factor.")
    ap.add_argument("--p", type=float, default=8.0, help="Angular decay exponent.")
    ap.add_argument("--alpha", type=float, default=0.2, help="Distance penalty coefficient.")
    ap.add_argument("--beta", type=float, default=0.6, help="Resolution sensitivity exponent.")
    ap.add_argument("--zeta", type=float, default=1.0, help="Distance penalty exponent.")
    ap.add_argument("--fov-deg", type=float, default=45.0, help="Default half-FOV in degrees if per-row fov_deg is absent.")
    ap.add_argument("--samples", type=int, default=4096, help="Number of sphere samples for integration.")
    ap.add_argument("--seed", type=int, default=12345, help="Random seed for sphere sampling.")
    ap.add_argument("--save-coverage", type=str, default=None, help="Optional path to save per-sample coverage CSV.")
    args = ap.parse_args()

    res = style_error_from_csv(
        csv_path=args.obs,
        k=args.k, p=args.p, alpha=args.alpha, beta=args.beta, zeta=args.zeta,
        default_fov_deg=args.fov_deg,
        n_samples=args.samples, seed=args.seed,
        save_coverage=args.save_coverage,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
