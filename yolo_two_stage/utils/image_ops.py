from __future__ import annotations

from typing import Tuple

import numpy as np
import cv2


def clip_bbox_to_image(
	bbox_xyxy: Tuple[float, float, float, float], image_shape: Tuple[int, int, int]
) -> Tuple[int, int, int, int]:
	x1, y1, x2, y2 = bbox_xyxy
	height, width = image_shape[:2]
	x1_i = int(max(0, min(width - 1, round(x1))))
	y1_i = int(max(0, min(height - 1, round(y1))))
	x2_i = int(max(0, min(width - 1, round(x2))))
	y2_i = int(max(0, min(height - 1, round(y2))))
	if x2_i < x1_i:
		x1_i, x2_i = x2_i, x1_i
	if y2_i < y1_i:
		y1_i, y2_i = y2_i, y1_i
	return x1_i, y1_i, x2_i, y2_i


def crop_image(
	image_bgr: np.ndarray,
	bbox_xyxy: Tuple[float, float, float, float],
	pad_pixels: int = 0,
) -> np.ndarray:
	x1, y1, x2, y2 = bbox_xyxy
	x1 -= pad_pixels
	y1 -= pad_pixels
	x2 += pad_pixels
	y2 += pad_pixels

	x1_i, y1_i, x2_i, y2_i = clip_bbox_to_image((x1, y1, x2, y2), image_bgr.shape)
	return image_bgr[y1_i : y2_i + 1, x1_i : x2_i + 1].copy()
