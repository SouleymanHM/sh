#!/usr/bin/env python3
import sys, os
from pathlib import Path
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tifffile as tiff
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree
from typing import List, Tuple, Optional, Dict
from astropy.stats import mad_std

LENSLET_PITCH = 30    # in pixels
PIXEL_PITCH = 5.86    # in um
PUPIL_DIAMETER = 3    # in mm

def save_tiff_uint8(array: np.ndarray, path: str):
    """
    Save image as uint8 TIFF with contrast normalization.
    """
    array = array.astype(np.float32, copy=False)
    min_val, max_val = float(array.min()), float(array.max())
    if max_val > min_val:
        array_uint8 = ((array - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
    else:
        array_uint8 = np.zeros_like(array, dtype=np.uint8)
    tiff.imwrite(str(path), array_uint8, compression="deflate")



def denoise_with_blank(
    img: np.ndarray,
    blank: np.ndarray,
    mode: str = "global",       # "global" (one a,b) or "tiled" (per-block a,b)
    tile_size: int = 128,       # used when mode="tiled"
    clip: bool = True,          # clip result to original dtype range
    trim_quantile: float = 0.01 # robustify regression by trimming extremes (1% tails)
):
    """
    Remove fixed-pattern noise using a blank/noise reference frame.

    The model is:  clean = img - (a * blank + b)
    where (a, b) are estimated to best match the noise level (robustly).

    Args:
        img:   2D (or 3D single-channel) image (any numeric dtype).
        blank: Same shape as img; a blank/noise frame (e.g., dark frame).
        mode:  "global" uses a single affine (a,b). "tiled" estimates (a,b) per tile.
        tile_size: Size of square tiles for "tiled" mode.
        clip:  If True, clip output to the valid range of img's dtype.
        trim_quantile: Fraction of low/high tails trimmed from regression (robustness).

    Returns:
        denoised: float32 array by default (or clipped to original dtype range if clip=True).
        meta: dict with keys:
              - 'mode': "global" or "tiled"
              - 'a', 'b' for global mode
              - 'a_map', 'b_map' for tiled mode (float32 maps at tile resolution)
              - 'trim_quantile'
    """
    if img.shape != blank.shape:
        raise ValueError("img and blank must have the same shape")

    # Work in float32 to avoid precision/overflow issues.
    img_f   = np.asarray(img,   dtype=np.float32)
    blank_f = np.asarray(blank, dtype=np.float32)

    # Helper: robust affine fit y ≈ a*x + b using trimmed least squares
    def robust_affine_fit(y, x, q=trim_quantile):
        yv = y.reshape(-1)
        xv = x.reshape(-1)

        # Remove NaNs/Infs
        mask = np.isfinite(yv) & np.isfinite(xv)
        yv, xv = yv[mask], xv[mask]
        if yv.size < 10:
            return 1.0, 0.0  # fallback

        # Trim extremes (winsorization by masking)
        lo_y, hi_y = np.quantile(yv, [q, 1 - q])
        lo_x, hi_x = np.quantile(xv, [q, 1 - q])
        m2 = (yv >= lo_y) & (yv <= hi_y) & (xv >= lo_x) & (xv <= hi_x)
        yv, xv = yv[m2], xv[m2]
        if yv.size < 10:
            return 1.0, 0.0

        # Closed-form LS for a,b
        x_mean = float(xv.mean())
        y_mean = float(yv.mean())
        x_var  = float(((xv - x_mean) ** 2).mean())
        if x_var <= 1e-12:
            return 1.0, y_mean  # degenerate: blank nearly constant

        cov_xy = float(((xv - x_mean) * (yv - y_mean)).mean())
        a = cov_xy / x_var
        b = y_mean - a * x_mean
        return a, b

    meta = {"mode": mode, "trim_quantile": float(trim_quantile)}

    if mode == "global":
        a, b = robust_affine_fit(img_f, blank_f)
        denoised_f = img_f - (a * blank_f + b)
        meta.update({"a": float(a), "b": float(b)})

    elif mode == "tiled":
        H, W = img_f.shape[:2]
        a_map = np.zeros(( (H + tile_size - 1)//tile_size,
                           (W + tile_size - 1)//tile_size), dtype=np.float32)
        b_map = np.zeros_like(a_map)

        denoised_f = np.empty_like(img_f)
        for ti, y0 in enumerate(range(0, H, tile_size)):
            for tj, x0 in enumerate(range(0, W, tile_size)):
                y1 = min(y0 + tile_size, H)
                x1 = min(x0 + tile_size, W)
                patch_img   = img_f[y0:y1, x0:x1]
                patch_blank = blank_f[y0:y1, x0:x1]

                a, b = robust_affine_fit(patch_img, patch_blank)
                a_map[ti, tj] = a
                b_map[ti, tj] = b

                denoised_f[y0:y1, x0:x1] = patch_img - (a * patch_blank + b)

        meta.update({"a_map": a_map, "b_map": b_map})
    else:
        raise ValueError('mode must be "global" or "tiled"')

    # Optional clipping to input dtype range
    if clip:
        # Determine valid range from original dtype (assume integer if int type, else no clip)
        if np.issubdtype(img.dtype, np.integer):
            info = np.iinfo(img.dtype)
            denoised_f = np.clip(denoised_f, info.min, info.max)
        # For floats, typically you don't clip; keep as-is

    return denoised_f.astype(np.float32, copy=False), meta





def centroid(img: np.ndarray):
    image = img.astype(np.float32)
    intensity_sum = image.sum()
    if intensity_sum <= 0:
        return float("nan"), float("nan")
    yy, xx = np.indices(image.shape, dtype=np.float32)
    centroid_x = float((image * xx).sum() / intensity_sum)
    centroid_y = float((image * yy).sum() / intensity_sum)
    return (centroid_x, centroid_y)


def overlay_frame_center_red(image: np.ndarray, center_x: float, center_y: float, radius: int = 6):
    visualization = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
    visualization = cv2.cvtColor(visualization, cv2.COLOR_GRAY2BGR)
    cv2.circle(visualization, (int(round(center_x)), int(round(center_y))), radius, (0, 0, 255), -1, lineType=cv2.LINE_AA)
    return visualization


def shift_image_to_center(img: np.ndarray, centroid_x: float, centroid_y: float) -> np.ndarray:
    image_height, image_width = img.shape
    frame_center_x = (image_width - 1) / 2.0
    frame_center_y = (image_height - 1) / 2.0

    shift_x = frame_center_x - centroid_x
    shift_y = frame_center_y - centroid_y

    transformation_matrix = np.array([[1, 0, shift_x], [0, 1, shift_y]], dtype=np.float32)

    return cv2.warpAffine(
        img.astype(np.float32),
        transformation_matrix,
        (image_width, image_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )


def gaussian_blur(img: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    image = img.astype(np.float32)
    blurred_image = gaussian_filter(image, sigma=sigma)
    return blurred_image


def image_spread(img: np.ndarray, centroid_x: float, centroid_y: float) -> float:
    """
    Calculate the intensity-weighted spread (RMS radius) of a 2D grayscale image,
    relative to a given centroid (centroid_x, centroid_y).

    Spread = sqrt( sum(I * d^2) / sum(I) )
    where d^2 is squared distance from (centroid_x, centroid_y).
    """
    # Ensure float for math
    img = img.astype(np.float32)

    # Build coordinate grids
    y, x = np.indices(img.shape)

    # Squared distances from provided centroid
    d2 = (x - centroid_x) ** 2 + (y - centroid_y) ** 2

    # Intensity-weighted mean squared distance
    total_intensity = img.sum()
    if total_intensity <= 0:
        return 0.0

    spread = np.sqrt((img * d2).sum() / total_intensity)
    return spread




def shift_image_by_offset(image: np.ndarray, shift_x: float, shift_y: float) -> np.ndarray:
    
    shift_y = -1 * shift_y
    
    """
    Shift the input image by (shift_x, shift_y) using subpixel-accurate affine transformation.

    Parameters:
        image    : 2D numpy array (grayscale image)
        shift_x  : Horizontal shift in pixels (positive = right, negative = left)
        shift_y  : Vertical shift in pixels (positive = down, negative = up)

    Returns:
        Shifted image of the same shape and dtype (float32)
    """
    image = image.astype(np.float32, copy=False)
    image_height, image_width = image.shape

    # Construct affine translation matrix
    translation_matrix = np.float32([
        [1, 0, shift_x],
        [0, 1, shift_y]
    ])

    # Apply affine warp with linear interpolation and black border padding
    shifted_image = cv2.warpAffine(
        image,
        translation_matrix,
        (image_width, image_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0
    )

    return shifted_image



def quad_center(img: np.ndarray, max_iterations=10, ksize=3):

    img = img.astype(np.float32, copy=False)
    image_intensity = float(img.sum())

    image_height, image_width = img.shape
    #print("shape", img.shape)

    frame_center_y = image_height // 2
    frame_center_x = image_width // 2

    if not np.isfinite(image_intensity) or image_intensity <= 0.0:
        return (img, frame_center_x, frame_center_y)




    if max_iterations > 10:
        max_iterations = 10

    while max_iterations:
        #start = time.time()
        # Quadrants
        UL = img[0:frame_center_y, 0:frame_center_x]  # Upper Left
        UR = img[0:frame_center_y, frame_center_x:]   # Upper Right
        LL = img[frame_center_y:, 0:frame_center_x]   # Lower Left
        LR = img[frame_center_y:, frame_center_x:]    # Lower Right

        # Intensity sums
        UL_sum = float(UL.sum())
        UR_sum = float(UR.sum())
        LL_sum = float(LL.sum())
        LR_sum = float(LR.sum())

        total_intensity = float(img.sum())

        # Horizontal: (Left - Right)
        left_sum = UL_sum + LL_sum
        right_sum = UR_sum + LR_sum

        # Vertical: (Bottom - Top)
        bottom_sum = LL_sum + LR_sum
        top_sum = UL_sum + UR_sum

        gain = 0.3  # Unitless gain
        
        h_error = (left_sum - right_sum) / total_intensity
        v_error = (bottom_sum - top_sum) / total_intensity
        
        #print("h error: ",h_error)
        #print("v error: ",v_error)

        horizontal_shift = gain * image_width * h_error
        vertical_shift   = gain * image_height * v_error  # Flip for correct direction

        # Clip shifts
        cap = 0.2 * min(image_width, image_height)
        horizontal_shift = float(np.clip(horizontal_shift, -cap, cap))
        vertical_shift   = float(np.clip(vertical_shift,   -cap, cap))

        # Apply shift
        img = shift_image_by_offset(img, horizontal_shift, vertical_shift)

        #print(max_iterations, (horizontal_shift, vertical_shift),
        #      (np.sqrt(h_error ** 2 + v_error ** 2)))
        #print()

        # Visualization
        #plt.figure(figsize=(6, 6))
        #plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        #plt.title(f"Iteration {max_iterations}")
        #plt.axis('off')
        #plt.show()

        max_iterations -= 1

        if (np.sqrt(h_error ** 2 + v_error ** 2)) <= 0.01:
            #elapsed = time.time() - start
            #print(f"Elapsed time: {elapsed:.3f} seconds")
            break

    # Return the last computed shift center relative to original

    centroid_x, centroid_y = centroid(gaussian_blur(img,3))  # Not used for shift

    return (img, centroid_x, centroid_y)



def get_pattern_radius(image: np.ndarray, centroid_x, centroid_y):
    image_height, image_width = image.shape
    cy = int(round(centroid_y))
    cx = int(round(centroid_x))

    rays = {}

    # Horizontal
    rays["left"]  = image[cy, 0:cx+1][::-1]
    rays["right"] = image[cy, cx:image_width]

    # Vertical
    rays["up"]   = image[0:cy+1, cx][::-1]
    rays["down"] = image[cy:image_height, cx]

    # Diagonal ↘ (down-right)
    rays["down_right"] = np.diagonal(image[cy:, cx:])

    # Diagonal ↖ (up-left)
    rays["up_left"] = np.diagonal(image[:cy+1, :cx+1])[::-1]

    # Diagonal ↗ (up-right)
    rays["up_right"] = np.diagonal(np.fliplr(image[:cy+1, cx:]))[::-1]

    # Diagonal ↙ (down-left)
    rays["down_left"] = np.diagonal(np.fliplr(image[cy:, :cx+1]))

    ordered_ray_names = [
        "left", "right", "up", "down",
        "up_left", "up_right", "down_left", "down_right"
    ]

    ray_centers = []

    for ray_name in ordered_ray_names:
        ray_values = np.asarray(rays[ray_name], dtype=np.float64)
        total_intensity = ray_values.sum()
        if total_intensity <= 0.0:
            ray_centers.append((float(centroid_x), float(centroid_y)))
            continue

        pixel_indices = np.arange(ray_values.size, dtype=np.float64)
        offset = (ray_values * pixel_indices).sum() / total_intensity

        if ray_name == "left":
            ray_centers.append((centroid_x - offset, centroid_y))
        elif ray_name == "right":
            ray_centers.append((centroid_x + offset, centroid_y))
        elif ray_name == "up":
            ray_centers.append((centroid_x, centroid_y - offset))
        elif ray_name == "down":
            ray_centers.append((centroid_x, centroid_y + offset))
        elif ray_name == "up_left":
            ray_centers.append((centroid_x - offset, centroid_y - offset))
        elif ray_name == "up_right":
            ray_centers.append((centroid_x + offset, centroid_y - offset))
        elif ray_name == "down_left":
            ray_centers.append((centroid_x - offset, centroid_y + offset))
        elif ray_name == "down_right":
            ray_centers.append((centroid_x + offset, centroid_y + offset))
    
    ray_centers = np.asarray(ray_centers, dtype=np.float64)

    mean_x = float(np.mean(ray_centers[:, 0]))
    mean_y = float(np.mean(ray_centers[:, 1]))
    
    mean_x_adjustment = 1.0
    mean_x *= mean_x_adjustment

    

    radius = float(2 * np.sqrt((mean_x - centroid_x) ** 2 + (mean_y - centroid_y) ** 2))

    return (mean_x, mean_y, radius)


def crop_square(image: np.ndarray,
                center_x: float,
                center_y: float,
                side_length: int,
                pad_to_size: bool = False,
                pad_value: float = 0.0) -> np.ndarray:
    """
    Crop a square patch of given side length centered at (center_x, center_y).

    - If pad_to_size=False (default): the crop is clipped at image borders and may be smaller than side_length.
    - If pad_to_size=True: the crop is padded with pad_value to always return (side_length x side_length).

    Returns the cropped (and possibly padded) image patch.
    """
    image_height, image_width = image.shape[:2]

    # Round center to nearest pixel index
    center_x_index = int(round(center_x))
    center_y_index = int(round(center_y))

    # Compute nominal crop bounds [x0:x1), [y0:y1)
    half = side_length // 2
    x0_nominal = center_x_index - half
    y0_nominal = center_y_index - half
    x1_nominal = x0_nominal + side_length
    y1_nominal = y0_nominal + side_length

    # Clip to image bounds
    x0 = int(round(max(0, x0_nominal)))
    y0 = int(round(max(0, y0_nominal)))
    x1 = int(round(min(image_width,  x1_nominal)))
    y1 = int(round(min(image_height, y1_nominal)))

    cropped = image[y0:y1, x0:x1]

    if not pad_to_size:
        return cropped

    # Pad to exact (side_length x side_length) if needed
    pad_top    = max(0, -y0_nominal)
    pad_left   = max(0, -x0_nominal)
    pad_bottom = max(0, y1_nominal - image_height)
    pad_right  = max(0, x1_nominal - image_width)

    if cropped.ndim == 2:
        pad_widths = ((pad_top, pad_bottom), (pad_left, pad_right))
    else:
        # For (H, W, C) images, do not pad channels
        pad_widths = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))

    return np.pad(cropped, pad_widths, mode="constant", constant_values=pad_value)


def find_spots(img: np.ndarray, max_spots=1000, quality=0.01, min_dist=20):
    """Detect bright spots using Shi–Tomasi corner detection."""
    img_u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    pts = cv2.goodFeaturesToTrack(img_u8,
                                  maxCorners=max_spots,
                                  qualityLevel=quality,
                                  minDistance=min_dist,
                                  blockSize=5,
                                  useHarrisDetector=False)
    if pts is None:
        return []
    return [(float(x), float(y)) for [[x, y]] in pts]





def remove_spacing_outliers(spots, tol=3.0, min_pts_per_line=4, cluster_tol=10.0):
    """
    Remove 'dead-pixel' or misaligned spots based on inconsistent spacing along clustered lines.

    Args:
        spots: array-like [(x, y), ...] of detected spot coordinates.
        tol: how many MADs from the median spacing counts as an outlier (default 3).
        min_pts_per_line: minimum spots required to analyze a line.
        cluster_tol: tolerance (pixels) for line clustering along x and y (passed to cluster_lines).

    Returns:
        filtered_spots: list of (x, y) with inconsistent line-spacing outliers removed.
        outliers: list of (x, y) that were removed.
    """
    from math import isnan
    from copy import deepcopy

    def cluster_lines(values: np.ndarray, tol: float = 10.0):
        values = np.asarray(values, float)
        order = np.argsort(values)
        v = values[order]
        centers = []
        idx_map = np.full(len(values), -1, dtype=int)
        if len(v) == 0:
            return np.array([]), idx_map
        start = 0
        lid = 0
        for i in range(1, len(v)):
            if abs(v[i] - v[i - 1]) > tol:
                seg = v[start:i]
                centers.append(seg.mean())
                idx_map[order[start:i]] = lid
                start = i
                lid += 1
        seg = v[start:]
        centers.append(seg.mean())
        idx_map[order[start:]] = lid
        centers = np.array(centers, float)
        sort_order = np.argsort(centers)
        inv = np.argsort(sort_order)
        return centers[sort_order], inv[idx_map]

    spots = np.asarray(spots, dtype=np.float64)
    if len(spots) < 3:
        return spots, []

    # --- cluster into approximate rows and columns ---
    x, y = spots[:, 0], spots[:, 1]
    x_centers, col_ids = cluster_lines(x, tol=cluster_tol)
    y_centers, row_ids = cluster_lines(y, tol=cluster_tol)

    # function for robust MAD
    def mad(arr):
        med = np.median(arr)
        return 1.4826 * np.median(np.abs(arr - med))

    keep_mask = np.ones(len(spots), bool)
    outliers = []

    # --- check rows ---
    for row in np.unique(row_ids):
        mask = row_ids == row
        if mask.sum() < min_pts_per_line:
            continue
        line = spots[mask]
        line = line[np.argsort(line[:, 0])]  # sort along x
        diffs = np.diff(line[:, 0])
        med_d = np.median(diffs)
        mad_d = mad(diffs)
        if mad_d == 0 or np.isnan(mad_d):
            continue
        # an outlier has two large adjacent diffs or one tiny one breaking pattern
        bad = np.where((diffs < med_d - tol * mad_d) | (diffs > med_d + tol * mad_d))[0]
        # remove the middle spot if both adjacent diffs are abnormal
        for b in bad:
            if 0 < b < len(line) - 1:
                outlier = line[b]
                idx = np.where((spots == outlier).all(axis=1))[0]
                keep_mask[idx] = False

    # --- check columns ---
    for col in np.unique(col_ids):
        mask = col_ids == col
        if mask.sum() < min_pts_per_line:
            continue
        line = spots[mask]
        line = line[np.argsort(line[:, 1])]  # sort along y
        diffs = np.diff(line[:, 1])
        med_d = np.median(diffs)
        mad_d = mad(diffs)
        if mad_d == 0 or np.isnan(mad_d):
            continue
        bad = np.where((diffs < med_d - tol * mad_d) | (diffs > med_d + tol * mad_d))[0]
        for b in bad:
            if 0 < b < len(line) - 1:
                outlier = line[b]
                idx = np.where((spots == outlier).all(axis=1))[0]
                keep_mask[idx] = False

    filtered = spots[keep_mask]
    outliers = spots[~keep_mask]
    return [filtered.tolist(), outliers.tolist()]




def draw_spots_on_image(
    pts: List[Tuple[float, float]],
    shape: Tuple[int, int],
    radius: int = 2,
    intensity: int = 255
    ) -> np.ndarray:
    """
    Construct a grayscale image from a list of (x, y) points.
    
    Args:
        pts: List of (x, y) coordinates (can be float; will be rounded).
        shape: (height, width) of the output image.
        radius: Radius of each dot (in pixels).
        intensity: Intensity of the dots (0–255).
    
    Returns:
        Grayscale image (uint8) with white dots on black background.
    """
    img = np.zeros(shape, dtype=np.uint8)

    for x, y in pts:
        cv2.circle(img, center=(int(round(x)), int(round(y))),
                   radius=radius, color=intensity, thickness=-1)

    return img







def sigma_clip_spots(spots, sigma=2.0, max_iter=2):
    """
    Iterative sigma clipping on spots based on nearest neighbor distances.
    Keeps only spots whose distances to neighbors are within a tolerance.
    """
    if not spots:
        return []

    pts = np.array(spots, dtype=np.float64)
    keep_mask = np.ones(len(pts), dtype=bool)

    for it in range(max_iter):
        pts_kept = pts[keep_mask]
        if len(pts_kept) < 10:
            break

        # pairwise distances
        dists = np.linalg.norm(
            pts_kept[None, :, :] - pts_kept[:, None, :],
            axis=-1
        )
        np.fill_diagonal(dists, np.inf)

        # get 8 nearest neighbors for each point
        nearest = np.sort(dists, axis=1)[:, :8]
        d_min = nearest.min(axis=1)
        d_max = nearest.max(axis=1)

        # stats for min
        mu_min, std_min = d_min.mean(), d_min.std()
        # stats for max
        mu_max, std_max = d_max.mean(), d_max.std()

        # clipping mask
        mask = (
            (d_min > mu_min - sigma*std_min) & (d_min < mu_min + sigma*std_min) &
            (d_max > mu_max - sigma*std_max) & (d_max < mu_max + sigma*std_max)
        )

        keep_mask[keep_mask] = mask

        #print(f"Iter {it+1}: kept {mask.sum()}/{len(mask)} spots | "
        #      f"avg min dist={mu_min:.2f}, avg max dist={mu_max:.2f}")

        if mask.all():
        #if mask.all() or abs((mu_max + mu_min) * 0.5 - 52) < 1:
            break

    return [tuple(pt) for pt in pts[keep_mask]]





def sigma_clip_spots_mad(spots,
                         sigma=2.5,
                         max_iter=5,
                         k_neighbors=5,         # (2) use median of k-NN distances
                         max_drop_frac=0.10):   # (6) cap removals per iteration (e.g., 10%)
    """
    Iterative robust clipping of spots using the median of k-NN distances and MAD.
    Only a capped fraction of the worst outliers are removed per iteration.

    Args:
        spots: list[(x, y)] of spot coordinates.
        sigma: robust z-score cutoff (on median k-NN distance).
        max_iter: maximum iterations.
        k_neighbors: number of nearest neighbors whose distances are summarized (median).
        max_drop_frac: max fraction of currently-kept spots to drop per iteration (0..1).

    Returns:
        List[(x, y)] of kept spots.
    """
    if not spots:
        return []

    pts = np.array(spots, dtype=np.float64)
    keep_mask = np.ones(len(pts), dtype=bool)

    # Local robust scale (MAD) helper: consistent with Gaussian σ via 1.4826 factor.
    def _mad_std(x):
        x = np.asarray(x, dtype=np.float64)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        # Avoid zero division: fall back to small epsilon if extremely tight
        return 1.4826 * mad if mad > 0 else 1e-12, med  # returns (scale, center)

    for _ in range(max_iter):
        idx_kept = np.flatnonzero(keep_mask)
        pts_kept = pts[idx_kept]
        n = len(pts_kept)
        if n < max(k_neighbors + 2, 5):  # need enough points to form neighborhoods
            break

        # Pairwise distances among kept points (n x n)
        dists = np.linalg.norm(pts_kept[:, None, :] - pts_kept[None, :, :], axis=-1)
        np.fill_diagonal(dists, np.inf)

        # Sort neighbor distances for each point; take the k smallest
        k_eff = min(k_neighbors, n - 1)
        nearest_sorted = np.sort(dists, axis=1)[:, :k_eff]

        # (2) Feature: median of k-NN distances (stable vs min/max & edges)
        feat = np.median(nearest_sorted, axis=1)

        # Robust location/scale via MAD
        scale, center = _mad_std(feat)

        # Robust z-scores; points beyond |z| > sigma are candidates to drop
        z = (feat - center) / scale
        outlier_mask = np.abs(z) > sigma
        num_outliers = int(outlier_mask.sum())

        if num_outliers == 0:
            break  # nothing to drop

        # (6) Cap removals per iteration: only drop the worst few
        max_drops = max(1, int(np.ceil(max_drop_frac * n)))
        if num_outliers > max_drops:
            # get indices of candidates sorted by |z| descending; keep only top max_drops
            cand_idx = np.flatnonzero(outlier_mask)
            worst_order = np.argsort(np.abs(z[cand_idx]))[::-1]
            drop_local = cand_idx[worst_order[:max_drops]]
        else:
            drop_local = np.flatnonzero(outlier_mask)

        # Map local kept indices back to global indices
        drop_global = idx_kept[drop_local]

        # Update keep mask
        keep_mask[drop_global] = False

        # If we didn't actually drop anything (shouldn't happen), stop
        if not np.any(~keep_mask[idx_kept]):
            break

    return [tuple(pt) for pt in pts[keep_mask]]






def cluster_lines(values: np.ndarray, tol: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster 1D coordinates (x or y) into lines within tolerance 'tol'.

    Returns:
        centers: (L,) sorted array of line centers (float).
        line_indices: (N,) array assigning each input value to a line index (0..L-1).
    """
    values = np.asarray(values, float)
    order = np.argsort(values)
    v = values[order]

    centers = []
    idx_map = np.full(len(values), -1, dtype=int)

    if len(v) == 0:
        return np.array([]), idx_map

    start = 0
    lid = 0
    for i in range(1, len(v)):
        if abs(v[i] - v[i-1]) > tol:
            seg = v[start:i]
            centers.append(seg.mean())
            # map back to original indices
            for j in range(start, i):
                orig = np.where(values == v[j])[0][0]
                idx_map[orig] = lid
            start = i
            lid += 1

    # flush last segment
    seg = v[start:]
    centers.append(seg.mean())
    for j in range(start, len(v)):
        orig = np.where(values == v[j])[0][0]
        idx_map[orig] = lid

    centers = np.array(centers, float)
    # sort centers and remap indices to sorted order
    sort_order = np.argsort(centers)
    centers_sorted = centers[sort_order]
    inv = np.argsort(sort_order)  # old_id -> new_id
    line_indices = inv[idx_map]
    return centers_sorted, line_indices


def true_centroid(
    spots: List[Tuple[float, float]],
    tol: float = 10.0
    ) -> List[List[object]]:
    """
    Compute possible true centroids by clustering vertical and horizontal lines.
    Rule:
      - EVEN line count → single middle line
      - ODD line count  → two middle candidates
    Combine → up to 4 centroid hypotheses.

    Returns:
        A list of [spots_array, (cx, cy)] for each hypothesis.
    """
    pts = np.asarray(spots, float)
    if pts.size == 0:
        return []

    xs, ys = pts[:, 0], pts[:, 1]
    x_centers, _ = cluster_lines(xs, tol=tol)
    y_centers, _ = cluster_lines(ys, tol=tol)

    Lx, Ly = len(x_centers), len(y_centers)
    if Lx == 0 or Ly == 0:
        return []

    def middle_candidates(L: int) -> List[int]:
        if L % 2 == 0:
            return [L // 2]  # single middle line
        else:
            mid = L // 2
            return [mid - 1, mid] if mid - 1 >= 0 else [0, min(1, L - 1)]

    x_cands = middle_candidates(Lx)
    y_cands = middle_candidates(Ly)

    results: List[List[object]] = []
    for yi in y_cands:
        for xi in x_cands:
            cx, cy = x_centers[xi], y_centers[yi]
            results.append([pts.copy(), (float(cx), float(cy))])

    return results




def average_spot_spacing(spots, k=4):
    """
    Calculate the average spacing between spots.
    Uses k nearest neighbors for each spot (default=4).
    
    Args:
        spots: list of (x, y) coordinates
        k: number of nearest neighbors to consider

    Returns:
        avg_spacing: float, mean nearest-neighbor distance
    """
    if spots.size == 0:
        return 0.0

    pts = np.array(spots, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return float("nan")

    # pairwise distance matrix
    dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)  # ignore self-distance

    # get nearest k neighbors for each point
    nearest = np.sort(dists, axis=1)[:, :k]

    # mean over all points and neighbors
    avg_spacing = nearest.mean()
    return avg_spacing

def make_reference_spots(
    shape=(1200, 1920),
    spacing=LENSLET_PITCH
):
    """
    Generate a list of reference spot coordinates on a regular grid.

    Args:
        shape: (height, width) of the frame.
        spacing: distance (in pixels) between adjacent spots.

    Returns:
        spots: list of (x, y) coordinates (floats).
    """
    h, w = shape
    spots = [
        (float(x), float(y))
        for y in range(spacing // 2, h, spacing)
        for x in range(spacing // 2, w, spacing)
    ]

     # compute centroid
    if spots:
        xs, ys = zip(*spots)
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
    else:
        cx, cy = float("nan"), float("nan")

    return spots, (cx, cy)

def zernike_to_sph_cyl_axis(
    z4: float, z5: float, z6: float,
    k_defocus: float = 1.0,
    k_astig:   float = 1.0
) -> Tuple[float, float, float]:
    """
    Convert (z4,z5,z6) to ophthalmic prescription (Sphere, Cylinder, Axis).

    Mapping assumes:
      M   = k_defocus * z4
      J0  = k_astig   * z6
      J45 = k_astig   * z5

    Power-vector relations (Thibos convention):
      J0  = -(C/2) * cos(2*axis)
      J45 = -(C/2) * sin(2*axis)
      M   = S + C/2

    Returns:
      S (sphere), C (cylinder), axis_deg (in [0, 180))
    """
    # Zernike -> power vector
    M   = k_defocus * float(z4)
    J0  = k_astig   * float(z6)
    J45 = k_astig   * float(z5)

    # Cylinder magnitude (note the minus sign per Thibos)
    R = np.hypot(J0, J45)          # sqrt(J0^2 + J45^2)
    C = -2.0 * R

    # Axis in degrees (0..180)
    axis_rad = 0.5 * np.arctan2(J45, J0)
    axis_deg = np.degrees(axis_rad) % 180.0

    # Sphere
    S = M - 0.5 * C

    return float(S), float(C), float(axis_deg)


reference_spots, reference_centroid = make_reference_spots(shape=(1200,1920),spacing=LENSLET_PITCH)


def draw_spot_comparison(ref_spots, cap_spots, shape, 
                         ref_color=(0,255,0), cap_color=(0,0,255),
                         radius=2, thickness=-1,
                         ref_centroid=None, cap_centroid=None):
    """
    Draw reference and captured spots on the same image.

    Args:
        ref_spots: array-like [(x, y), ...] for reference (will be drawn GREEN)
        cap_spots: array-like [(x, y), ...] for captured (will be drawn RED)
        shape: (H, W) shape of the output image
        ref_color: BGR tuple for reference spots (default green)
        cap_color: BGR tuple for captured spots (default red)
        radius: circle radius for each spot
        thickness: circle thickness (-1 = filled)
        ref_centroid, cap_centroid: optional (cx, cy) to mark centers

    Returns:
        BGR image with both spot sets drawn.
    """
    # base blank or gray background
    if len(shape) == 2:
        img = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    else:
        img = np.zeros(shape, dtype=np.uint8)
    img[:] = (40, 40, 40)

    ref_spots = np.asarray(ref_spots, dtype=np.float32)
    cap_spots = np.asarray(cap_spots, dtype=np.float32)

    # Draw reference spots (green)
    for (x, y) in ref_spots:
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(img, (int(round(x)), int(round(y))), radius, ref_color, thickness)

    # Draw captured spots (red)
    for (x, y) in cap_spots:
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(img, (int(round(x)), int(round(y))), radius, cap_color, thickness)

    # Optional centroid markers
    if ref_centroid is not None:
        cx, cy = ref_centroid
        cv2.drawMarker(img, (int(round(cx)), int(round(cy))), (0,255,0),
                       markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)
    if cap_centroid is not None:
        cx, cy = cap_centroid
        cv2.drawMarker(img, (int(round(cx)), int(round(cy))), (0,0,255),
                       markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=1)

    return img
