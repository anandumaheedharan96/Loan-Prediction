from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np


def _put_label(image: np.ndarray, text: str, org: Tuple[int, int], color: Tuple[int, int, int]) -> None:
	font = cv2.FONT_HERSHEY_SIMPLEX
	scale = 0.5
	thickness = 1
	(text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
	x, y = org
	bg_top_left = (x, max(0, y - text_h - baseline - 2))
	bg_bottom_right = (x + text_w + 4, y)
	cv2.rectangle(image, bg_top_left, bg_bottom_right, color, thickness=-1)
	cv2.putText(image, text, (x + 2, y - 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_detections(image_bgr: np.ndarray, labeled_boxes: Iterable[object]) -> np.ndarray:
	output = image_bgr.copy()
	for lb in labeled_boxes:
		x1, y1, x2, y2 = lb.bbox_xyxy
		x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
		color = getattr(lb, 'color', (0, 255, 0))
		label = getattr(lb, 'label', '')
		score = getattr(lb, 'score', None)
		text = f"{label} {score:.2f}" if isinstance(score, (float, int)) else f"{label}"
		cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness=2)
		_put_label(output, text, (x1, y1), color)
	return output
