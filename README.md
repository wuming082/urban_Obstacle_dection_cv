# 🏙️ UrbanEye — 城市无障碍视觉感知系统

> **一个面向城市公共安全的轻量化障碍物检测系统**  
> 支持井盖缺失、盲道占用、路面坑洞、斑马线、红绿灯等关键目标的实时识别，  
> 兼具 **端侧嵌入式部署** 与 **云端服务化能力**，助力智慧城市与无障碍出行。

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-purple)](LICENSE)

---

## 🎯 项目目标

为视障人士、老年人及城市管理者提供**实时、低延迟、高精度**的城市道路障碍物感知能力：
- ✅ 识别 **井盖缺失/移位**
- ✅ 检测 **盲道占用**（车辆、杂物）
- ✅ 定位 **路面坑洞/破损**
- ✅ 识别 **斑马线** 与 **交通信号灯**

系统支持两种使用模式：
- **端侧模式**：在 Jetson Nano / 树莓派等设备上本地运行，无需联网
- **云端模式**：通过 WebSocket 或 REST API 提供服务，支持多设备接入

---

## ⚙️ 技术亮点

- **端云协同架构**：轻量模型部署于边缘，复杂任务交由云端
- **模型自动适配**：根据硬件（CPU/GPU）自动加载最优推理后端
- **双协议支持**：
  - 📡 **WebSocket**：实时视频流低延迟推理（`ws://:8821/ws/detect`）
  - 📤 **REST API**：单图上传分析（`POST /predict`）
- **配置驱动**：所有参数（模型路径、端口、阈值）通过 `config.yaml` 管理
- **容器化部署**：Docker 支持一键部署，适配 x86_64 / ARM64 环境
- **模型轻量化**：支持导出 ONNX / TFLite，便于嵌入式部署

---

