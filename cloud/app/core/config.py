# cloud/app/core/config.py
import yaml
from pathlib import Path
from typing import Literal, Optional
import torch

class ModelConfig:
    def __init__(self, data: dict):
        self.path = data.get("path", "yolov8n.pt")
        self.device = data.get("device", "auto")
        self.conf_threshold = float(data.get("conf_threshold", 0.4))
        self.iou_threshold = float(data.get("iou_threshold", 0.5))

    def get_device(self) -> str:
        """根据配置自动选择设备"""
        if self.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        else:
            return self.device

class ServerConfig:
    def __init__(self, data: dict):
        self.host: str = data.get("host", "127.0.0.1")
        self.port: int = data.get("port", 8821)
        self.reload: bool = data.get("reload", False)

class LoggingConfig:
    def __init__(self, data: dict):
        self.level: str = data.get("level", "INFO")

class Config:
    def __init__(self, config_path: str = "cloud/config/config.yaml"):
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(config_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.server = ServerConfig(data.get("server", {}))
        self.model = ModelConfig(data.get("model", {}))
        self.logging = LoggingConfig(data.get("logging", {}))

# 全局单例（懒加载）
_config: Optional[Config] = None

def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config