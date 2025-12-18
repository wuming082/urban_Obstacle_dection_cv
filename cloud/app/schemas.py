# cloud/app/schemas.py
from pydantic import BaseModel
from typing import List, Optional

class Detection(BaseModel):
    bbox: List[int]          # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    label: str

class PredictionResponse(BaseModel):
    detections: List[Detection]
    image_size: List[int]    # [width, height]