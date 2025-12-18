# minimal_inference.py
from ultralytics import YOLO
import cv2

# === 配置 ===
MODEL_PATH = "../../models/large_docker_models/yolo_v12_n/yolov12n.pt"        # 替换为你自己的模型路径，如 "models/best.pt"
INPUT_IMAGE = "picture_test/detection_form.jpg"        # 输入图片路径（可换成 URL 或本地路径）
OUTPUT_IMAGE = "output.jpg"      # 输出结果保存路径

# === 加载模型 ===
print("Loading model...")
model = YOLO(MODEL_PATH)

# === 推理 ===
print("Running inference...")
results = model(INPUT_IMAGE)

# === 可视化结果（YOLOv8 内置支持）===
# results[0].plot() 返回带框的 BGR numpy array (H, W, C)
annotated_frame = results[0].plot()

# === 保存结果 ===
cv2.imwrite(OUTPUT_IMAGE, annotated_frame)
print(f"Result saved to {OUTPUT_IMAGE}")

# （可选）显示图片（需 GUI 环境）
# cv2.imshow("Detection", annotated_frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()