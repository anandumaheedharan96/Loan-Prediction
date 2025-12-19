from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union, Optional

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
	bbox_xyxy: Tuple[float, float, float, float]
	score: float
	class_id: int
	class_name: str


class YOLODetector:
	def __init__(
		self,
		weights_path: str,
		device: Optional[str] = None,
		conf: float = 0.25,
		iou: float = 0.45,
	) -> None:
		self.model = YOLO(weights_path)
		self.device = device
		self.conf = conf
		self.iou = iou

	@property
	def class_names(self) -> dict:
		return self.model.names

	def predict(
		self, images: Union[np.ndarray, Sequence[np.ndarray]]
	) -> List[List[Detection]]:
		if isinstance(images, np.ndarray):
			images = [images]

		results = self.model.predict(
			list(images), conf=self.conf, iou=self.iou, device=self.device, verbose=False
		)

		parsed: List[List[Detection]] = []
		for result in results:
			boxes = result.boxes
			names = result.names
			if boxes is None or boxes.shape[0] == 0:
				parsed.append([])
				continue

			xyxy = boxes.xyxy.cpu().numpy()
			confs = boxes.conf.cpu().numpy()
			clss = boxes.cls.cpu().numpy().astype(int)

			dets: List[Detection] = []
			for (x1, y1, x2, y2), score, cls_id in zip(xyxy, confs, clss):
				class_name = names.get(int(cls_id), str(int(cls_id)))
				dets.append(
					Detection(
						bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
						score=float(score),
						class_id=int(cls_id),
						class_name=class_name,
					)
				)
			parsed.append(dets)
		return parsed

	def predict_single(self, image: np.ndarray) -> List[Detection]:
		return self.predict(image)[0]
