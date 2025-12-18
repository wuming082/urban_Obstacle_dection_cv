# cloud/app/core/model_loader.py
from ultralytics import YOLO
from pathlib import Path
import torch
import numpy as np
from .config import get_config

# 全局变量（模块级单例）
_model = None
_model_type = None  # 记录当前加载的是哪种模型

class YOLOWrapper:
    def __init__(self, model_path: str, device: str):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.device = device
        self.class_names = self.model.names

    def predict(self, img: np.ndarray):
        """img: BGR numpy array (H, W, C)"""
        results = self.model(img, verbose=False)
        result = results[0]
        detections = []
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = self.class_names[cls_id]
            detections.append({
                "bbox": xyxy.tolist(),
                "confidence": round(conf, 2),
                "class_id": cls_id,
                "label": label
            })
        return detections

    def __init__(self, model_path: str, device: str):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.device = device
        self.class_names = self.model.names

    def predict(self, img: np.ndarray):
        """img: BGR numpy array (H, W, C)"""
        results = self.model(img, verbose=False)
        result = results[0]
        detections = []
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = self.class_names[cls_id]
            detections.append({
                "bbox": xyxy.tolist(),
                "confidence": round(conf, 2),
                "class_id": cls_id,
                "label": label
            })
        return detections


class MMDetWrapper:
    def __init__(self, config_path: str, checkpoint_path: str, device: str):
        from mmdet.apis import init_detector
        self.model = init_detector(config_path, checkpoint_path, device=device)
        # 兼容新旧版本 class_names
        if hasattr(self.model, 'dataset_meta') and 'classes' in self.model.dataset_meta:
            self.class_names = self.model.dataset_meta['classes']
        else:
            self.class_names = self.model.CLASSES if hasattr(self.model, 'CLASSES') else []

    def predict(self, img: np.ndarray):
        """img: BGR numpy array (H, W, C)"""
        from mmdet.apis import inference_detector
        result = inference_detector(self.model, img)
        
        # MMDet v3.0+ 返回 DetDataSample
        bboxes = result.pred_instances.bboxes.cpu().numpy()
        scores = result.pred_instances.scores.cpu().numpy()
        labels = result.pred_instances.labels.cpu().numpy()

        detections = []
        for i in range(len(bboxes)):
            x1, y1, x2, y2 = bboxes[i]
            conf = float(scores[i])
            cls_id = int(labels[i])
            label = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": round(conf, 2),
                "class_id": cls_id,
                "label": label
            })
        return detections


def get_model():
    global _model
    if _model is None:
        config = get_config()
        model_type = config.model.type
        device = config.model.get_device()

        if model_type == "yolov12":
            model_path = Path(config.model.yolov12.path)
            if not model_path.exists():
                raise FileNotFoundError(f"YOLOv12 model not found: {model_path}")
            print(f"Loading YOLOv12 model from {model_path} on {device}...")
            _model = YOLOWrapper(str(model_path), device)

        elif model_type == "mmdet":
            cfg_path = Path(config.model.mmdet.config)
            ckpt_path = Path(config.model.mmdet.checkpoint)
            if not cfg_path.exists():
                raise FileNotFoundError(f"MMDet config not found: {cfg_path}")
            if not ckpt_path.exists():
                raise FileNotFoundError(f"MMDet checkpoint not found: {ckpt_path}")
            print(f"Loading MMDet model from {cfg_path} on {device}...")
            _model = MMDetWrapper(str(cfg_path), str(ckpt_path), device)

        else:
            raise ValueError(f"Unsupported model.type in config: '{model_type}'. Use 'yolov8' or 'mmdet'.")

    return _model