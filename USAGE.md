# UrbanEye 使用教学文档

## 项目概述

UrbanEye 是一个面向城市公共安全的轻量化障碍物检测系统，支持井盖缺失、盲道占用、路面坑洞、斑马线、红绿灯等关键目标的实时识别。系统支持端侧嵌入式部署和云端服务化能力。

## 安装步骤

### 环境要求
- Python 3.8+
- 支持的操作系统：Linux, macOS, Windows

### 安装依赖

1. 克隆项目仓库：
   ```bash
   git clone <repository-url>
   cd urban_Obstacle_dection_cv
   ```

2. 安装云端依赖：
   ```bash
   cd cloud
   pip install -r requirements.txt
   ```

3. 安装边缘依赖（如果需要）：
   - 对于边缘设备，可能需要额外的依赖，如 OpenCV, PyTorch 等。

## 配置

项目使用 YAML 配置文件管理参数。主要配置文件位于 `cloud/config/config.yaml`。

示例配置：
```yaml
# cloud/config/config.yaml
server:
  host: "0.0.0.0"
  port: 8821
  reload: false

model:
  # MMDetection 配置

  type: "yolov12" # 或 "mmdet"

  mmdet:
    config: "/home/dreamsky/GitStock/urban_Obstacle_dection_cv/models/small_edge_models/mobilenet_v2_500e/config.py"
    checkpoint: "/home/dreamsky/GitStock/urban_Obstacle_dection_cv/models/small_edge_models/mobilenet_v2_500e/epoch_500.pth"

  # yolo系列的配置
  yolov12:
    path: "/home/dreamsky/GitStock/urban_Obstacle_dection_cv/models/large_docker_models/yolo_v12_n/yolov12n.pt" # ← 替换为你自己的模型路径

  device: "auto" # "auto", "cpu", "cuda", "mps"
  conf_threshold: 0.4 # 置信度阈值
  iou_threshold: 0.5 # NMS IOU 阈值

logging:
  level: "INFO"
```

## 使用方法

### 云端模式

#### 启动服务器

进入 cloud/ 目录并运行：
```bash
python run_server.py
```
**这一步很重要，因为要先启动server服务器，才能使用后面的测试脚本**
服务器将在配置的端口启动（默认 8000）。

#### API 使用

- **REST API**: 发送 POST 请求到 `/predict` 端点上传图片进行检测。
- **WebSocket**: 连接到 `ws://localhost:8000/ws/detect` 进行实时视频流检测。

#### 示例请求

使用 python脚本 发送图片：
`/tests/request_test/picture_request_test.py`

使用时，在命令行内使用`python picture_request_test.py `以启动脚本 
**tips：需要在当前**`test/request_test/` **目录下才能正常运行该脚本**

### 边缘模式

边缘模式适用于嵌入式设备，如 ESP32S3 或树莓派。

#### 运行推理

使用 `edge/inference.py` 脚本进行本地推理（需要实现该脚本）。

示例代码：
```python
# 加载模型并进行推理
from ultralytics import YOLO
import cv2

model = YOLO("models/large_docker_models/yolo_v12_n/yolov12n.pt")
results = model("input_image.jpg")
annotated_frame = results[0].plot()
cv2.imwrite("output.jpg", annotated_frame)
```

## 测试

### 模型测试

运行 `tests/model_test/test.py` 来测试模型推理：
```bash
cd tests/model_test
python test.py
```

这将加载模型，对测试图片进行检测，并保存结果。

### 图片请求测试

（如果实现）运行 `tests/picture_test/picture_request_test.py` 来测试 API 请求。

## 模型

- **大模型**: YOLOv12n，适用于云端高精度检测。
- **小模型**: MobileNet v2，经过 500 轮训练，适用于边缘设备。

模型文件位于 `models/` 目录。

## 部署

### 边缘设备部署

对于边缘设备，需要导出模型为 ONNX 或 TFLite 格式，然后部署到设备上。

## 故障排除

- 确保所有依赖已安装。
- 检查配置文件路径是否正确。
- 对于 GPU 推理，确保 PyTorch 支持 CUDA。
- 如果遇到端口冲突，更改配置文件中的端口。

## 贡献

欢迎提交 issue 和 pull request 来改进项目。

## 许可证

MIT License