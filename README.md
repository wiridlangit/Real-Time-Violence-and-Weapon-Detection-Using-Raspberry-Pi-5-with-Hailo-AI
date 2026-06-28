# Real-Time Violence and Weapon Detection Using Raspberry Pi 5 with Hailo AI

## Overview

This repository contains the implementation of a real-time violence and weapon detection system developed for intelligent surveillance applications. The system leverages **YOLOv11** for object detection and activity recognition, while utilizing **Raspberry Pi 5** integrated with the **Hailo AI Accelerator** to achieve efficient edge inference with low latency.

The proposed system is capable of detecting:

* Violence/Fighting
* Knives
* Guns

When a suspicious event is detected, the system can generate real-time alerts, making it suitable for public safety, smart surveillance, and edge AI applications.

---

## Features

* Real-time violence detection
* Real-time weapon detection (Knife & Gun)
* Optimized for Raspberry Pi 5
* Hailo AI acceleration for fast inference
* YOLOv11-based deep learning model
* Lightweight edge deployment

---

## Hardware

* Raspberry Pi 5
* Hailo AI Accelerator (Hailo-8/Hailo-8L)
* CCTV
* Power Supply

---

## Software

* Python 3.10
* YOLOv11
* OpenCV
* HailoRT
* NumPy

---

## Usage

Run real-time detection:

```bash
yolo predict model=runs/detect/train/weights/(model_name).pt source=0
```

Run in Raspberry Pi:
```bash
python 
detector_draft4.py --labels-json violence-sajam-labels.json --hef-path Violence_Sajam_Detection.hef -i /dev/video0.
```
---

## Results

The system is designed to perform real-time inference on Raspberry Pi 5 using the Hailo AI accelerator, enabling efficient detection of violent activities and dangerous weapons while maintaining low latency suitable for edge surveillance applications.

---

## Publication

This project is based on our published research paper:

**Real-Time Violence and Weapon Detection Using Raspberry Pi 5 with Hailo AI**

IEEE Xplore:
https://ieeexplore.ieee.org/document/11369525

If you use this repository in your research, please consider citing our work.

---

## Citation

```bibtex
@ARTICLE{11369525,
  title={Real-Time Violence and Weapon Detection Using Raspberry Pi 5 with Hailo AI},
  journal={IEEE},
  year={2025},
  url={https://ieeexplore.ieee.org/document/11369525}
}
```

---

## License

This project is intended for research and educational purposes.

---

## Author

**Wiridlangit Jiwangga**
