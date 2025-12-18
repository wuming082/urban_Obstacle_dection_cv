# cloud/app/api/predict.py
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from io import BytesIO
from ..core.model_loader import get_model
from ..schemas import PredictionResponse, Detection

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    # 1. 读取上传的图片
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image")

    h, w = img.shape[:2]

    # 2. 获取单例模型并推理
    model = get_model()
    detections = model.predict(img)  # 抽象调用

    # 3. 转换为 Pydantic 模型（detections 已是标准 dict list）
    detection_objects = [
        Detection(
            bbox=det["bbox"],
            confidence=det["confidence"],
            class_id=det["class_id"],
            label=det["label"]
        )
        for det in detections
    ]

    return PredictionResponse(
        detections=detection_objects,
        image_size=[w, h]
    )