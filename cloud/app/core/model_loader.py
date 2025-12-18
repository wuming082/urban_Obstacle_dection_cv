# cloud/app/core/model_loader.py
from ultralytics import YOLO
from pathlib import Path
from .config import get_config

# 全局变量（模块级单例）
_model = None

def get_model():
    global _model
    if _model is None:
        config = get_config()
        model_path = Path(config.model.path)  # ← 替换为你的模型路径
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        device = config.model.get_device()
        print(f"Loading model from {model_path} on device: {device}...")
        
        print(f"Loading model from {model_path}...")
        _model = YOLO(str(model_path))
    return _model