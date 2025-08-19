from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np

from yolo_two_stage.detectors.yolo_detector import YOLODetector, Detection
from yolo_two_stage.utils.image_ops import crop_image
from yolo_two_stage.viz.annotator import draw_detections


@dataclass
class LabeledBox:
	bbox_xyxy: Tuple[float, float, float, float]
	label: str
	score: float
	color: Tuple[int, int, int]


class TwoStagePipeline:
	def __init__(
		self,
		stage1_weights_path: str,
		stage2_weights_path: str,
		vehicle_label_stage1: str = "vehicle",
		living_label_stage1: str = "living_beings",
		device: Optional[str] = None,
		stage1_conf: float = 0.25,
		stage2_conf: float = 0.25,
		stage1_iou: float = 0.45,
		stage2_iou: float = 0.45,
		crop_pad_pixels: int = 4,
	) -> None:
		self.stage1 = YOLODetector(
			stage1_weights_path, device=device, conf=stage1_conf, iou=stage1_iou
		)
		self.stage2 = YOLODetector(
			stage2_weights_path, device=device, conf=stage2_conf, iou=stage2_iou
		)
		self.vehicle_label_stage1 = vehicle_label_stage1
		self.living_label_stage1 = living_label_stage1
		self.crop_pad_pixels = crop_pad_pixels

	def _separate_stage1(self, detections: List[Detection]):
		vehicle_detections: List[Detection] = []
		living_detections: List[Detection] = []
		other_detections: List[Detection] = []
		for det in detections:
			if det.class_name == self.vehicle_label_stage1:
				vehicle_detections.append(det)
			elif det.class_name == self.living_label_stage1:
				living_detections.append(det)
			else:
				other_detections.append(det)
		return vehicle_detections, living_detections, other_detections

	def _classify_vehicles(
		self, image_bgr: np.ndarray, vehicle_detections: List[Detection]
	) -> List[str]:
		if not vehicle_detections:
			return []
		crops = [
			crop_image(image_bgr, det.bbox_xyxy, pad_pixels=self.crop_pad_pixels)
			for det in vehicle_detections
		]
		batch_results = self.stage2.predict(crops)
		labels: List[str] = []
		for dets in batch_results:
			if not dets:
				labels.append("unknown")
				continue
			top = max(dets, key=lambda d: d.score)
			labels.append(top.class_name)
		return labels

	def run_on_image(self, image_path: str, output_path: str) -> None:
		image_bgr = cv2.imread(image_path)
		if image_bgr is None:
			raise FileNotFoundError(f"Could not read image: {image_path}")

		stage1_detections = self.stage1.predict_single(image_bgr)
		vehicle_detections, living_detections, _ = self._separate_stage1(stage1_detections)

		vehicle_labels = self._classify_vehicles(image_bgr, vehicle_detections)

		labeled_boxes: List[LabeledBox] = []
		for det, vlabel in zip(vehicle_detections, vehicle_labels):
			labeled_boxes.append(
				LabeledBox(
					bbox_xyxy=det.bbox_xyxy,
					label=f"{vlabel}",
					score=det.score,
					color=(0, 153, 255),
				)
			)
		for det in living_detections:
			labeled_boxes.append(
				LabeledBox(
					bbox_xyxy=det.bbox_xyxy,
					label=f"{det.class_name}",
					score=det.score,
					color=(203, 62, 255),
				)
			)

		annotated = draw_detections(image_bgr, labeled_boxes)
		ok = cv2.imwrite(output_path, annotated)
		if not ok:
			raise RuntimeError(f"Failed to write output image: {output_path}")
