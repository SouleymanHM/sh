#!/usr/bin/env python3
import sys, os
from pathlib import Path
import time
import numpy as np
import cv2
import tifffile as tiff
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree


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
       
        img = overlay_frame_center_red(img, centroid_x, centroid_y)
        return img
    
    else:
        shift_image_to_center(img, (centroid_x + horizontal_shift), (centroid_y + vertical_shift))
        return quad_center(img, max_iterations)



folder_in = "tiff_folder"
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
        
    image = quad_center(image,5)

    # Build output filename (strip extension, add suffix, save to out/)
    base, ext = os.path.splitext(file)
    filename = f"{base}_quad_center{ext}"
    out_path = os.path.join(folder_out, filename)

   # Save result
    save_tiff_uint8(image, out_path)
