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

