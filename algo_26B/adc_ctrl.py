#a new version of the algorithm, designed to run faster and be more robust to noise.
#code for new method of moments
import numpy as np
import matplotlib.pyplot as plt
from hcipy import *
from scipy import ndimage

def moment_angle(
    crop: np.ndarray,
    fwhm_px: float = 6.0,
    max_iter: int = 5,
    tol_deg: float = 0.05,
) -> float:

    # 0. Physical Positivity Enforcer (Fix for pre-subtracted negative pixels)
    # Floor raw input to zero immediately so negative background noise cannot act as "negative mass"
    crop_pos = np.maximum(crop.astype(np.float64), 0.0)
    ny, nx = crop_pos.shape
    y_grid, x_grid = np.mgrid[0:ny, 0:nx]

    # 1. Estimate Residual Noise Floor on positive-floored signal
    border_pixels = np.concatenate(
        [
            crop_pos[0, :],
            crop_pos[-1, :],
            crop_pos[:, 0],
            crop_pos[:, -1],
        ]
    )

    # Estimate noise level using robust MAD on non-zero background
    bg_level = np.median(border_pixels)
    mad = np.median(np.abs(border_pixels - bg_level))
    sigma_noise = 1.4826 * mad if mad > 0 else 1

    # 2. Subtract residual local background and re-enforce non-negativity
    img_sub = np.maximum(crop_pos - bg_level, 0.0)

    # 2. Smooth to find initial 2D Peak & Estimate Peak SNR
    sigma_smooth = fwhm_px / 2.355
    smoothed = ndimage.gaussian_filter(np.maximum(img_sub, 0.0), sigma=sigma_smooth)
    peak_y, peak_x = np.unravel_index(np.argmax(smoothed), smoothed.shape)
    peak_val = img_sub[peak_y, peak_x]

    snr_est = peak_val / sigma_noise if sigma_noise > 0 else 100.0

    # 3. Dynamic SNR Adjustments
    if snr_est < 15.0:
        # Low SNR regime (5-15): Aggressive noise cut, tight window
        snr_thresh_factor = 2.5
        min_win_scale = 0.8
        max_win_scale = 1.2
    elif snr_est < 50.0:
        # Mid SNR regime (15-50): Balanced settings
        snr_thresh_factor = 1.5
        min_win_scale = 0.8
        max_win_scale = 1.8
    else:
        # High SNR regime (>50): Relaxed threshold, wide window for accuracy
        snr_thresh_factor = 1.0
        min_win_scale = 1.0
        max_win_scale = 2.5

    # Threshold image based on estimated SNR
    img_clean = np.maximum(
        img_sub - (snr_thresh_factor * sigma_noise), 0.0
    )

    # Initial state
    x_bar, y_bar = float(peak_x), float(peak_y)
    angle = 0.0
    sigma_u = fwhm_px * 0.5
    sigma_v = fwhm_px * 0.5

    # 4. Iterative Moment Computation
    for iteration in range(max_iter):
        prev_angle = angle

        dx = x_grid - x_bar
        dy = y_grid - y_bar

        # Check centroid drift: Don't let iteration drift > 1 FWHM from smoothed peak
        dist_from_peak = np.sqrt((x_bar - peak_x) ** 2 + (y_bar - peak_y) ** 2)
        if dist_from_peak > fwhm_px:
            x_bar, y_bar = float(peak_x), float(peak_y)
            dx = x_grid - x_bar
            dy = y_grid - y_bar

        # Rotate to principal axis frame
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        u = dx * cos_a + dy * sin_a
        v = -dx * sin_a + dy * cos_a

        # Soft Elliptical Gaussian Window
        window = np.exp(-0.5 * ((u / sigma_u) ** 2 + (v / sigma_v) ** 2))
        weights = img_clean * window
        total_weight = np.sum(weights)

        if total_weight == 0:
            # Fallback for extreme noise: use raw peak-centered unweighted moments
            weights = np.maximum(img_clean, 0.0)
            total_weight = np.sum(weights)
            if total_weight == 0:
                return 0.0

        # Refine Centroid
        x_bar = np.sum(x_grid * weights) / total_weight
        y_bar = np.sum(y_grid * weights) / total_weight

        # Compute 2nd Central Moments
        dx_c = x_grid - x_bar
        dy_c = y_grid - y_bar

        mu20 = np.sum((dx_c**2) * weights)
        mu02 = np.sum((dy_c**2) * weights)
        mu11 = np.sum((dx_c * dy_c) * weights)

        # Unbiased Angle Calculation
        angle = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)

        # Calculate shape variances
        var_u = 0.5 * (
            mu20 + mu02 + np.sqrt((mu20 - mu02) ** 2 + 4 * (mu11**2))
        )
        var_v = 0.5 * (
            mu20 + mu02 - np.sqrt((mu20 - mu02) ** 2 + 4 * (mu11**2))
        )

        # Adaptively update window dimensions with SNR-based bounds
        raw_sigma_u = np.sqrt(max(var_u / total_weight, 0.25))
        raw_sigma_v = np.sqrt(max(var_v / total_weight, 0.25))

        sigma_u = np.clip(
            raw_sigma_u, min_win_scale * fwhm_px, max_win_scale * fwhm_px
        )
        sigma_v = np.clip(
            raw_sigma_v, min_win_scale * fwhm_px, max_win_scale * fwhm_px
        )

        # Check convergence
        if (
            np.abs(np.degrees(angle - prev_angle)) < tol_deg
            and iteration > 0
        ):
            break

    return float(np.degrees(angle))
