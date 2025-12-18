# cloud/app/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
from .core.model_loader import get_model
from .core.config import get_config
from .api.predict import router as predict_router

# Lifespan：启动时预加载模型（可选，也可以懒加载）
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Preloading model...")
    config = get_config()
    print(f"   Server: {config.server.host}:{config.server.port}")
    get_model()  # 触发模型加载
    yield
    # Shutdown（可清理资源）

app = FastAPI(
    title="UrbanEye AI Server",
    description="Obstacle detection API for urban accessibility",
    lifespan=lifespan
)

# 路由注册
app.include_router(predict_router)
app.include_router(predict_router, prefix="/api/v1")  # 可选版本前缀

# WebSocket 保留（你之前的逻辑）
from fastapi import WebSocket
import cv2
import numpy as np

@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    await websocket.accept()
    model = get_model()  # 复用同一个模型实例
    try:
        while True:
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                await websocket.send_json({"error": "Invalid image"})
                continue

            results = model(img, verbose=False)
            result = results[0]

            detections = []
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                detections.append({
                    "bbox": xyxy.tolist(),
                    "confidence": round(conf, 2),
                    "class_id": cls_id,
                    "label": label
                })

            await websocket.send_json({"detections": detections})
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()