# test_single_image.py
import requests
import sys
import json
import cv2
import numpy as np
import os

# === 配置 ===
API_URL = "http://localhost:8821/predict"
IMAGE_PATH = "../picture_test/detection_form.jpg"
OUTPUT_VIS_IMAGE = "output_with_boxes.jpg"  # 可视化结果保存路径

def draw_boxes_on_image(image_path: str, detections: list, output_path: str):
    """在图片上绘制检测框和标签"""
    # 读取原始图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image for drawing: {image_path}")
        return

    # 颜色映射（按 class_id 区分颜色）
    colors = {}
    for det in detections:
        cls_id = det["class_id"]
        if cls_id not in colors:
            # 用 class_id 生成固定颜色（BGR）
            np.random.seed(cls_id)
            colors[cls_id] = np.random.randint(0, 255, size=3).tolist()

    # 绘制每个检测框
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]
        label = det["label"]
        cls_id = det["class_id"]

        color = colors[cls_id]
        # 画框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        # 画标签背景
        label_text = f"{label} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - 20), (x1 + w, y1), color, -1)
        # 写标签
        cv2.putText(image, label_text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 保存结果
    cv2.imwrite(output_path, image)
    print(f"Visualized image saved to: {os.path.abspath(output_path)}")

def test_prediction(image_path: str):
    try:
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
            print(f"Uploading {image_path} to {API_URL}...")
            response = requests.post(API_URL, files=files, timeout=10)

        if response.status_code == 200:
            result = response.json()
            print("Detection result received.")

            # 保存 JSON 结果（可选）
            with open("last_result.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            # 在本地绘制检测框
            draw_boxes_on_image(image_path, result["detections"], OUTPUT_VIS_IMAGE)

        else:
            print(f"Request failed: {response.status_code} - {response.text}")

    except FileNotFoundError:
        print(f"Image not found: {image_path}")
    except requests.exceptions.ConnectionError:
        print(f"Failed to connect to {API_URL}. Is the server running?")
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else IMAGE_PATH
    if not os.path.exists(img_path):
        print(f"Image path does not exist: {img_path}")
        sys.exit(1)
    test_prediction(img_path)