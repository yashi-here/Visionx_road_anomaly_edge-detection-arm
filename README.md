# 🚀 Real-Time Road Anomaly Detection on ARM Edge (Raspberry Pi 4)

An optimized **CPU-only edge AI system** for real-time pothole detection deployed on **Raspberry Pi 4 (ARM Cortex-A72)** using an **INT8-quantized YOLOv5n model**.

This project demonstrates full-stack edge AI engineering — from dataset preparation and GPU training to quantization, ARM optimization, and real-time system benchmarking.

---

## 📌 Project Highlights

* ✅ Lightweight YOLOv5n (1.76M parameters)
* ✅ INT8 Quantization (47% size reduction)
* ✅ Real-time CPU inference (13–16.6 FPS)
* ✅ Multi-threaded ARM optimization (4 threads)
* ✅ Thermal stability validation (46–66°C)
* ✅ Logging overhead benchmarking
* ✅ Resolution vs performance trade-off analysis
* ✅ Full telemetry monitoring (FPS, CPU, RAM, Temp)

---

# 🧠 1. Problem Statement

Road surface degradation causes vehicle damage, safety hazards, and maintenance inefficiencies.

This project implements a **low-power, CPU-only edge AI system** capable of detecting potholes in real-time on ARM-based hardware — without GPU or accelerators.

Designed for:

* Smart city monitoring
* Vehicle-mounted inspection systems
* Edge-based infrastructure analytics
* Sustainable AI deployment

---

# 📊 2. Model Summary

| Metric                       | Value     |
| ---------------------------- | --------- |
| Model                        | YOLOv5n   |
| Parameters                   | 1,760,518 |
| GFLOPs                       | 4.1       |
| Input Resolution             | 320×320   |
| mAP@0.5                      | 76.3%     |
| mAP@0.5:0.95                 | 45.3%     |
| Precision                    | 80.5%     |
| Optimal Confidence Threshold | 0.45      |
| IoU Threshold                | 0.45      |

---

# 📦 3. Dataset

* Total Images: **2602**
* Total Annotations: **9652**
* Classes: **1 (pothole)**

### Split

| Split      | Images |
| ---------- | ------ |
| Train      | 1845   |
| Validation | 507    |
| Test       | 250    |

Average potholes per image: ~3.7

Dataset Source: Roboflow
License: CC BY 4.0

---

# 🏗 4. Model Development Workflow

## Model Development & Deployment Workflow

![Model Workflow](diagrams/model_workflow.jpeg)

Steps:

1. Dataset Preparation
2. YOLOv5n Training (Tesla T4 GPU)
3. Model Evaluation
4. Export to TFLite
5. INT8 Quantization
6. Raspberry Pi Deployment
7. Resolution Optimization
8. Real-Time Benchmarking

---

# 🔬 5. Training Environment

### Hardware

* NVIDIA Tesla T4
* CUDA 13.0

### Software

* PyTorch 2.10.0+cu128
* YOLOv5
* Python 3.12

Training Configuration:

```bash
python train.py \
  --img 320 \
  --batch 16 \
  --epochs 100 \
  --patience 20 \
  --data data.yaml \
  --weights yolov5n.pt \
  --name pothole_fast
```

Training Time: ~55 minutes
Epochs: 100

---

# ⚙️ 6. Quantization & Model Compression

Export Command:

```bash
python export.py --weights best.pt --include tflite --int8 --img 320
```

| Format         | Size    |
| -------------- | ------- |
| FP32 (.pt)     | 3.66 MB |
| INT8 (.tflite) | 1.93 MB |

📉 **47% compression**

Benefits:

* Lower memory footprint
* Faster loading
* Reduced memory bandwidth
* Better ARM cache efficiency
* Integer arithmetic optimization

---

# 🖥 7. Edge Deployment (Raspberry Pi 4)

## End-to-End ARM Edge System

![ARM Edge System](diagrams/arm_edge_system.jpeg)

### Hardware

* Raspberry Pi 4
* ARM Cortex-A72 (Quad-core)
* Raspberry Pi OS (Debian Trixie 64-bit)
* Passive heat sink

### Software

* numpy 1.26.4
* opencv-python 4.13.0.92
* tflite-runtime 2.14.0

Inference:

* 4-thread TFLite interpreter
* CPU-only execution
* Headless capable

---

# 🔄 8. Real-Time Embedded Inference Pipeline

![Inference Pipeline](diagrams/inference_pipeline.jpeg)

Pipeline:

Camera (YUV420, 320×320)
→ Raw Frame Buffer
→ YUV → BGR Conversion
→ INT8 TFLite Inference (4 Threads)
→ Confidence Filtering (0.45)
→ Non-Maximum Suppression (IoU 0.45)
→ Bounding Box Rendering
→ Telemetry Logging

---

# 🚀 9. Real-Time Performance

## Resolution vs FPS Trade-off

![Performance Tradeoff](diagrams/performance_tradeoff.png)

| Input Size | FPS         |
| ---------- | ----------- |
| 416×416    | ~3 FPS      |
| 320×320    | 13–16.6 FPS |

Throughput improved by ~5× after resolution optimization.

---

## Latency

At 15 FPS:

≈ **66 ms per frame**

---

## Logging Overhead Analysis

| Mode                | FPS     |
| ------------------- | ------- |
| Bounding boxes only | 15–18 |
| With CSV logging    | 8–16    |

Demonstrates embedded I/O bottleneck awareness.

---

# 🌡 10. Thermal & System Stability

During sustained inference:

* Temperature range: **46°C – 66°C**
* No thermal throttling observed
* Stable multi-core CPU utilization
* Continuous operation validated

Telemetry logged:

* FPS
* CPU usage %
* RAM usage %
* Temperature
* Detection confidence
* Bounding box coordinates

Sample logs available in:

```
results/sample_detection.csv
results/sample_system_metrics.csv
```

---

# 📊 11. Validation Results

Stored in `results/`:

* PR_curve.png
* F1_curve.png
* P_curve.png
* R_curve.png
* val_batch*_pred.jpg
* val_batch*_labels.jpg

---

# 🛠 12. Installation (Edge Device)

```bash
sudo apt update
pip install -r requirements.txt
python edge/realtime_inference.py
```

---

# 🧪 13. Reproducing Training

Install:

```bash
pip install -r requirements_training.txt
```

Then run training command above.

---

# 📂 14. Repository Structure

```
models/
    best.pt
    best-int8.tflite

edge/
    realtime_inference.py
    video_inference.py

training/
    train_command.txt
    export_command.txt

results/
    PR_curve.png
    F1_curve.png
    sample_detection.csv
    sample_system_metrics.csv

diagrams/
    model_workflow.jpeg
    inference_pipeline.jpeg
    arm_edge_system.jpeg
    performance_tradeoff.png

data/
    data.yaml
```

---

# 🔮 15. Future Work

* Active cooling for outdoor roadside deployment
* ARM Ethos-U / NPU acceleration
* Distributed fleet-level monitoring
* Cloud aggregation for city-scale analytics
* Confidence calibration refinement

---

# 🏆 16. Conclusion

This project demonstrates that:

Carefully optimized lightweight models + INT8 quantization + system-level engineering
→ Enable stable real-time computer vision on low-power ARM CPUs
→ Without GPU acceleration

It balances:

* Accuracy (76.3% mAP@0.5)
* Efficiency (1.76M parameters)
* Compression (47% reduction)
* Real-time performance (~15 FPS)
* Thermal stability
* Sustainable CPU-only deployment

Delivering a deployable edge AI solution aligned with the goals of efficient and scalable AI on Arm-based SoCs.

---

# 📜 License

Dataset: CC BY 4.0
Code: MIT License

---
