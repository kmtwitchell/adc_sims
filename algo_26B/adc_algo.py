import numpy as np


def speckle_angle(
    image: np.ndarray,
    axis: str = "vertical",
    threshold_ratio: float = 0.2,
    core_ratio: float = 0.4
) -> float:
    """Measures the pointing direction of a single satellite spot.
    Parameters
    ----------
    image : np.ndarray
        2D science image array containing JUST the satellite spot.

    axis : str
        'vertical' if the spot is elongated vertically (slice along rows to find x-centers),
        'horizontal' if elongated horizontally (slice along columns to find y-centers).
    threshold_ratio : float
        Ratio of max-to-min intensity required to keep a slice (filters out background noise).
    core_ratio : float
        minimum ratio of the max of a slice to the global max of the image to be used in the fit

    Returns
    -------
    angle : float
        The fitted orientation angle in radians relative to the positive x-axis.
    x_coords : np.ndarray
        contains the x coordinates of the fitted centers.
    y_coords : np.ndarray
        contains the y coordinates of the fitted centers.
    """

    ny, nx = image.shape
    global_max = np.max(image)

    slice_indices = []
    centers = []

    if axis == "vertical":
        # Spot major axis is along Y (rows); cross-sections are horizontal (columns/rows)
        for row_idx in range(ny):
            slice_data = image[row_idx, :]
            s_min, s_max = np.min(slice_data), np.max(slice_data)

            # Quality check: ensure signal-to-noise in slice exceeds threshold ratio
            if s_max <= 0 or (s_max - s_min) / (s_max + 1e-8) < threshold_ratio:
                continue

            #quality check 2: to exclude airy core, make sure there's enough signal in the slice compared to the WHOLE image
            if s_max < core_ratio * global_max:
                continue

            # Isolate central core (truncate outside local minima around peak)
            peak_idx = np.argmax(slice_data)
            left = peak_idx
            while left > 0 and slice_data[left - 1] < slice_data[left]:
                left -= 1
            right = peak_idx
            while right < ny - 1 and slice_data[right + 1] < slice_data[right]:
                right += 1

            core_data = slice_data[left : right + 1]
            core_coords = np.arange(left, right + 1)

            # Sub-pixel center via 1st moment (Center of Mass)
            weights = np.maximum(core_data - s_min, 0)
            if np.sum(weights) > 0:
                x_center = np.sum(core_coords * weights) / np.sum(weights)
                slice_indices.append(row_idx)
                centers.append(x_center)

        y_coords = np.array(slice_indices)
        x_coords = np.array(centers)

        if len(x_coords) < 3:
            raise ValueError(
                "Insufficient valid slices to measure spot orientation."
            )


    else:  # horizontal orientation
        for col_idx in range(nx):
            slice_data = image[:, col_idx]
            s_min, s_max = np.min(slice_data), np.max(slice_data)

            if s_max <= 0 or (s_max - s_min) / (s_max + 1e-8) < threshold_ratio:
                continue

            if s_max < core_ratio * global_max:
                continue

            peak_idx = np.argmax(slice_data)
            left = peak_idx
            while left > 0 and slice_data[left - 1] < slice_data[left]:
                left -= 1
            right = peak_idx
            while right < nx - 1 and slice_data[right + 1] < slice_data[right]:
                right += 1

            core_data = slice_data[left : right + 1]
            core_coords = np.arange(left, right + 1)

            weights = np.maximum(core_data - s_min, 0)
            if np.sum(weights) > 0:
                y_center = np.sum(core_coords * weights) / np.sum(weights)
                slice_indices.append(col_idx)
                centers.append(y_center)

        x_coords = np.array(slice_indices)
        y_coords = np.array(centers)

        if len(y_coords) < 3:
            raise ValueError(
                "Insufficient valid slices to measure spot orientation."
            )

    return x_coords, y_coords


def spot_direction(
    x_coords: np.ndarray, y_coords: np.ndarray
) -> np.ndarray:
    """Finds the unit direction vector along a spot's major axis using SVD/PCA.

    Handles vertical, horizontal, and diagonal orientations without slope explosion.
    """
    # Stack and center coordinates around mean
    coords = np.column_stack([x_coords, y_coords])
    coords_centered = coords - np.mean(coords, axis=0)

    # SVD gives principal axes; first right-singular vector is the primary axis
    _, _, vh = np.linalg.svd(coords_centered)
    direction_vec = vh[0]  # Unit vector (v_x, v_y)

    return direction_vec  # Normalized vector [v_x, v_y]

def pair_offset(
    v1: np.ndarray, v2: np.ndarray
) -> float:
    """Calculates small differential angle between two spot vectors directly.

    No arctan, no division by zero, no angle wrapping bugs near 0 or 90 degrees.
    """
    # Ensure vectors point away from zero-order center (same radial sense)
    if np.dot(v1, v2) < 0:
        v2 = -v2

    # 2D Cross Product: v1_x * v2_y - v1_y * v2_x = sin(delta_theta) ~ delta_theta
    sin_dtheta = v1[0] * v2[1] - v1[1] * v2[0]

    return float(sin_dtheta)  # Offset angle in radians