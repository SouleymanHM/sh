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
PIXEL_PITCH = 5       # in um
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

def save_tiff(array: np.ndarray, path: str):
    tiff.imwrite(str(path), array)

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
        vertical_shift   = -1 * gain * image_height * v_error  # Flip for correct direction

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
import numpy as np

def sigma_clip_spots_mad_2(spots,
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



def sigma_clip_spots_mad(spots, sigma=2.5, max_iter=5):
    """
    Iterative MAD-based sigma clipping on spots based on nearest neighbor distances.
    Keeps only spots whose distances to neighbors are within a tolerance.
    """
    if not spots:
        return []

    pts = np.array(spots, dtype=np.float64)
    keep_mask = np.ones(len(pts), dtype=bool)

    for it in range(max_iter):
        pts_kept = pts[keep_mask]
        if len(pts_kept) < 5:
            break

        # pairwise distances
        dists = np.linalg.norm(
            pts_kept[None, :, :] - pts_kept[:, None, :],
            axis=-1
        )
        np.fill_diagonal(dists, np.inf)

        # get 4 nearest neighbors for each point
        nearest = np.sort(dists, axis=1)[:, :4]
        d_min = nearest.min(axis=1)
        d_max = nearest.max(axis=1)

        # robust stats using median and MAD
        med_min = np.median(d_min)
        mad_min = mad_std(d_min)

        med_max = np.median(d_max)
        mad_max = mad_std(d_max)

        # clipping mask
        mask = (
            (d_min > med_min - sigma * mad_min) & (d_min < med_min + sigma * mad_min) &
            (d_max > med_max - sigma * mad_max) & (d_max < med_max + sigma * mad_max)
        )

        keep_mask[keep_mask] = mask

        #print(f"Iter {it+1}: kept {mask.sum()}/{len(mask)} spots | "
        #      f"median min dist={med_min:.2f}, median max dist={med_max:.2f}")

        if mask.all():
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


#*************************************
def fit_zernike_4_6_from_displacements(obs: np.ndarray, ref: np.ndarray):
    """
    Least-squares fit of Z4–Z6 from displacement vectors.
    obs: Nx2 captured points (normalized)
    ref: Nx2 reference points (normalized)
    """
    if obs.shape != ref.shape or obs.shape[1] != 2:
        raise ValueError("obs and ref must be Nx2 arrays")

    dxdy = obs - ref
    dx, dy = dxdy[:, 0], dxdy[:, 1]
    x,  y  = ref[:, 0],  ref[:, 1]

    # Correct gradients for Z4–Z6 (unit pupil)
    Ax = np.stack([2.0*x, 2.0*y, 2.0*x], axis=1)   # d/dx for Z4, Z5, Z6
    Ay = np.stack([2.0*y, 2.0*x, -2.0*y], axis=1)  # d/dy for Z4, Z5, Z6

    A = np.vstack([Ax, Ay])
    b = np.hstack([dx, dy])

    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    z4, z5, z6 = coeffs.tolist()
    return z4, z5, z6

#*************************************


def zernike_from_nearest_with_centroids_1(
    reference_spots: List[Tuple[float, float]],
    captured_spots:  List[Tuple[float, float]],
    ref_centroid:    Tuple[float, float],
    cap_centroid:    Tuple[float, float],
    max_dist: float = np.inf,
    min_matches: int = 12,
    return_pairs: bool = False
):
    """
    Overlay by your provided centroids (translation only), then:
      - For each reference spot, find nearest captured spot.
      - Drop pairs with distance > max_dist (no interpolation).
      - Fit Z4–Z6 from surviving pairs.

    Args:
        reference_spots: [(x,y)] for the reference grid (any length >= 2)
        captured_spots:  [(x,y)] for the captured frame (any length >= 2)
        ref_centroid:    (cx_ref, cy_ref) centroid of reference_spots (you provide)
        cap_centroid:    (cx_cap, cy_cap) centroid of captured_spots (you provide)
        max_dist:        distance gate in pixels (≈ 1/3–1/2 of lenslet pitch is typical)
        min_matches:     minimum matched pairs required to attempt fit
        return_pairs:    if True, also return the matched arrays (ref_m, obs_m)

    Returns:
        (z4, z5, z6)  or  ((z4, z5, z6), ref_m, obs_m) if return_pairs=True

    Raises:
        RuntimeError if not enough pairs survive the distance gate.
    """
    ref = np.asarray(reference_spots, dtype=np.float64)
    obs = np.asarray(captured_spots,  dtype=np.float64)
    if ref.ndim != 2 or ref.shape[1] != 2 or obs.ndim != 2 or obs.shape[1] != 2:
        raise ValueError("Inputs must be arrays/lists of (x,y) points.")
    if len(ref) < 2 or len(obs) < 2:
        raise RuntimeError("Not enough points to match.")

    # 1) translate captured → reference using provided centroids
    refc = np.asarray(ref_centroid, dtype=np.float64)
    capc = np.asarray(cap_centroid, dtype=np.float64)
    if refc.shape != (2,) or capc.shape != (2,):
        raise ValueError("ref_centroid and cap_centroid must be length-2 tuples.")
    shift = refc - capc
    obs_shifted = obs + shift

    # 2) nearest-neighbor match: reference -> shifted captured
    tree = cKDTree(obs_shifted)
    d, idx = tree.query(ref, k=1)

    # 3) distance gate
    keep = d <= max_dist if np.isfinite(max_dist) else np.ones_like(d, dtype=bool)
    ref_m = ref[keep]
    obs_m = obs_shifted[idx[keep]]

    # 4) enough matches?
    if len(ref_m) < min_matches:
        raise RuntimeError(f"Insufficient matches after gating: {len(ref_m)} < {min_matches}")

    # 5) fit Z4–Z6 (uses your existing function)
    z4, z5, z6 = fit_zernike_4_6_from_displacements(obs_m, ref_m)

    if return_pairs:
        return (float(z4), float(z5), float(z6)), ref_m, obs_m
    return (float(z4), float(z5), float(z6))

#####################################
def zernike_from_nearest_with_centroids(
    reference_spots: List[Tuple[float, float]],
    captured_spots:  List[Tuple[float, float]],
    ref_centroid:    Tuple[float, float],
    cap_centroid:    Tuple[float, float],
    lenslet_pitch_px: float,
    pixel_pitch_um:   float,
    pupil_diameter_mm: float,
    max_dist: float = np.inf,
    min_matches: int = 12,
    return_pairs: bool = False
):
    """
    Overlay by your provided centroids (translation only), then:
      - For each reference spot, find nearest captured spot.
      - Drop pairs with distance > max_dist (no interpolation).
      - Fit Z4–Z6 from surviving pairs using normalized pupil coords.

    Units:
        - lenslet_pitch_px: lenslet pitch in pixels
        - pixel_pitch_um: camera pixel pitch in microns
        - pupil_diameter_mm: pupil diameter in millimeters

    Returns:
        (z4, z5, z6) or ((z4, z5, z6), ref_m, obs_m)
    """
    ref = np.asarray(reference_spots, dtype=np.float64)
    obs = np.asarray(captured_spots,  dtype=np.float64)
    if ref.ndim != 2 or ref.shape[1] != 2 or obs.ndim != 2 or obs.shape[1] != 2:
        raise ValueError("Inputs must be arrays/lists of (x,y) points.")
    if len(ref) < 2 or len(obs) < 2:
        raise RuntimeError("Not enough points to match.")

    # 1) translate captured → reference using provided centroids
    refc = np.asarray(ref_centroid, dtype=np.float64)
    capc = np.asarray(cap_centroid, dtype=np.float64)
    if refc.shape != (2,) or capc.shape != (2,):
        raise ValueError("ref_centroid and cap_centroid must be length-2 tuples.")
    shift = refc - capc
    obs_shifted = obs + shift

    # 2) nearest-neighbor match: reference -> shifted captured
    tree = cKDTree(obs_shifted)
    d, idx = tree.query(ref, k=1)

    # 3) distance gate
    keep = d <= max_dist if np.isfinite(max_dist) else np.ones_like(d, dtype=bool)
    ref_m = ref[keep]
    obs_m = obs_shifted[idx[keep]]

    if len(ref_m) < min_matches:
        raise RuntimeError(f"Insufficient matches after gating: {len(ref_m)} < {min_matches}")

    # 4) Convert to normalized pupil coordinates
    # Convert pixels → mm via lenslet pitch:
    #   lenslet_pitch_mm = lenslet_pitch_px * pixel_pitch_mm
    pixel_pitch_mm = pixel_pitch_um / 1000.0
    lenslet_pitch_mm = lenslet_pitch_px * pixel_pitch_mm
    scale = lenslet_pitch_mm  # mm per lenslet

    radius = pupil_diameter_mm / 2.0

    # Normalize pixel displacements to unit radius pupil
    ref_norm = ((ref_m - refc) * (pixel_pitch_um / 1000)) / (pupil_diameter_mm / 2)
    obs_norm = ((obs_m - refc) * (pixel_pitch_um / 1000)) / (pupil_diameter_mm / 2)


    # 5) fit Z4–Z6 in normalized coordinates
    z4, z5, z6 = fit_zernike_4_6_from_displacements(ref_norm, obs_norm)
    print("Pupil radius (pixels):", radius/pixel_pitch_mm)
    print("Mean displacement (pixels):", np.mean(np.linalg.norm(obs_m - ref_m, axis=1)))
    print("Mean displacement (normalized):", np.mean(np.linalg.norm(obs_norm - ref_norm, axis=1)))
    if return_pairs:
        return (float(z4), float(z5), float(z6)), ref_m, obs_m
    return (float(z4), float(z5), float(z6))

#######################################

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


def test_zernike_to_sph_cyl_axis(z4, z5, z6, k_defocus=1.0, k_astig=1.0):
    M   = k_defocus * float(z4)
    J0  = k_astig   * float(z6)
    J45 = k_astig   * float(z5)

    R = np.hypot(J0, J45)          # magnitude of astigmatism vector
    C = -2.0 * R                   # cylinder (Thibos negative cyl)
    axis_rad = 0.5 * np.arctan2(J45, J0)
    axis_deg = np.degrees(axis_rad) % 180.0
    S = M - 0.5 * C                # sphere = M - C/2

    return float(S), float(C), float(axis_deg)

reference_spots, reference_centroid = make_reference_spots(shape=(1200,1920),spacing=LENSLET_PITCH)

