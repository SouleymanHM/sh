#!/usr/bin/env python3
import sys, os
from pathlib import Path
import time
import numpy as np
import cv2
import tifffile as tiff
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree
from typing import List, Tuple
from astropy.stats import mad_std


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
    visualization = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
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
    img = img.astype(np.float64)

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


def quad_center(img: np.ndarray, max_iterations = 5):
    img = img.astype(np.float64, copy=False)
    blurred_img = gaussian_blur(img, 2)
    centroid_x, centroid_y = centroid(blurred_img)
    img = shift_image_to_center(img,centroid_x,centroid_y)

    image_height, image_width = img.shape
    #frame_center_y = image_height // 2   # row index of horizontal split
    #frame_center_x = image_width // 2    # column index of vertical split

    centroid_x = int(round(centroid_x))
    centroid_y = int(round(centroid_y))
    
    top_half    = img[0:centroid_y, :]               # rows 0 .. frame_center_y-1, all columns
    bottom_half = img[centroid_y:image_height, :]    # rows frame_center_y .. end, all columns
    left_half   = img[:, 0:centroid_x]               # all rows, cols 0 .. frame_center_x-1
    right_half  = img[:, centroid_x:image_width]     # all rows, cols frame_center_x .. end
    
    image_intensity = float(img.sum())
    top_half_intensity = float(top_half.sum())
    bottom_half_intensity = float(bottom_half.sum())
    left_half_intensity = float(left_half.sum())
    right_half_intensity = float(right_half.sum())
    
    if not np.isfinite(image_intensity) or image_intensity <= 0.0:
        # Nothing to do; keep current image
        return img

    spread = image_spread(img, centroid_x, centroid_y)
    spread = float(spread)

    gain = 1.5 * spread

    horizontal_shift = gain * (left_half_intensity - right_half_intensity) / image_intensity
    vertical_shift = gain * (bottom_half_intensity - top_half_intensity) / image_intensity

    cap = 0.5 * min(image_width, image_height)
    horizontal_shift = float(np.clip(horizontal_shift, -cap, cap))
    vertical_shift   = float(np.clip(vertical_shift,   -cap, cap))
    
    shift_image_to_center(img, (centroid_x + horizontal_shift), (centroid_y + vertical_shift))

    print(max_iterations, (horizontal_shift, vertical_shift), (np.sqrt(horizontal_shift ** 2 + vertical_shift ** 2)/spread))

    max_iterations -= 1
    
    if max_iterations <= 0 or (np.sqrt(horizontal_shift ** 2 + vertical_shift ** 2)/spread) <= 0.05:
        print()
        transformation_matrix = np.array([[1, 0, horizontal_shift], [0, 1, vertical_shift]], dtype=np.float32)
        img = cv2.warpAffine(
            img,
            transformation_matrix,
            (image_width, image_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0
        )
       
        return (img, centroid_x, centroid_y)
    
    else:
        shift_image_to_center(img, (centroid_x + horizontal_shift), (centroid_y + vertical_shift))
        return quad_center(img, max_iterations)

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

    

    magnitude = float(np.sqrt(mean_x ** 2 + mean_y ** 2))

    return (mean_x, mean_y, magnitude)


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


def find_spots(img: np.ndarray, max_spots=1000, quality=0.01, min_dist=15):
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







def sigma_clip_spots(spots, sigma=2.5, max_iter=5):
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

        print(f"Iter {it+1}: kept {mask.sum()}/{len(mask)} spots | "
              f"avg min dist={mu_min:.2f}, avg max dist={mu_max:.2f}")

        if mask.all():
        #if mask.all() or abs((mu_max + mu_min) * 0.5 - 52) < 1:
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

        print(f"Iter {it+1}: kept {mask.sum()}/{len(mask)} spots | "
              f"median min dist={med_min:.2f}, median max dist={med_max:.2f}")

        if mask.all():
            break

    return [tuple(pt) for pt in pts[keep_mask]]






folder_in = "shm"
folder_out = "out"
os.makedirs(folder_out, exist_ok=True)  # make sure "out/" exists

for file in os.listdir(folder_in):
    if not (file.lower().endswith(".tif") or file.lower().endswith(".tiff")):
        continue  # skip non-TIFF files

    # Full path to input file
    file_path = os.path.join(folder_in, file)

    # Load image
    image = tiff.imread(file_path)

    # Process image
        
    out = quad_center(image,5)
    image = out[0]
    ray_centers = get_pattern_radius(image, out[1], out[2])

    #image = overlay_frame_center_red(image, ray_centers[0], ray_centers[1])
    image = crop_square(image,out[1],out[2],ray_centers[2])
    shape = image.shape
    spots = find_spots(image) 

    spots = sigma_clip_spots(spots, 2.5, 2)
    #spots = sigma_clip_spots_mad(spots, 2, 2)
    
    print(len(spots))
    
    print()

    image = draw_spots_on_image(spots, shape)

    # Build output filename (strip extension, add suffix, save to out/)
    base, ext = os.path.splitext(file)
    filename = f"{base}_quad_center{ext}"
    out_path = os.path.join(folder_out, filename)

   # Save result
    save_tiff_uint8(image, out_path)
