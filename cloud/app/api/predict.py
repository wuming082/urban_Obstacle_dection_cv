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
    results = model(img, verbose=False)
    result = results[0]

    # 3. 构建检测结果
    detections = []
    for box in result.boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        detections.append(
            Detection(
                bbox=xyxy.tolist(),
                confidence=round(conf, 2),
                class_id=cls_id,
                label=label
            )
        )

    return PredictionResponse(
        detections=detections,
        image_size=[w, h]
    )