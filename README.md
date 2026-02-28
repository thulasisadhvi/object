
# Construction Site Safety Detection System

This project is a production-ready custom object detection system designed to identify safety gear (helmets, vests, gloves, etc.) in real-time. It features a fine-tuned **YOLOv8** model, a **FastAPI** REST interface, and is fully containerized using **Docker**.

## 🚀 Quick Start (Docker)

Ensure you have Docker Desktop running, then execute the following commands in your terminal:

1. **Build the image:**
```bash
docker build -t construction-safety-app .

```


2. **Run the container:**
```bash
docker run -p 8000:8000 construction-safety-app

```


3. **Access the API:**
Open `http://localhost:8000/docs` in your browser to test the model using the interactive Swagger UI.

---

## 🏗️ System Architecture

* **Model:** YOLOv8 Nano (Fine-tuned for 25 epochs).
* **Dataset:** Site Construction Safety dataset (Roboflow) containing 7 classes (Helmet, Vest, Person, etc.).
* **Framework:** Ultralytics YOLO & FastAPI.
* **Inference:** Accepts image uploads via a POST request and returns JSON bounding box coordinates and confidence scores.

---

## 📊 Performance Metrics

The model was evaluated on a dedicated test set with the following results:

* **mAP@0.5:** 0.9343 (93.4%)
* **mAP@0.5:0.95:** 0.7271 (72.7%)
* **Inference Speed:** ~4.8ms per image (on T4 GPU).

---

## 📁 Project Structure

```text
├── main.py              # FastAPI application script
├── weights/
│   └── best.pt          # Fine-tuned model weights
├── requirements.txt     # Python dependencies
├── Dockerfile           # Multi-stage Docker configuration
├── .dockerignore        # Optimized for build speed and size
└── submission.yml       # Automation workflow definition

```

---

## 🛠️ Implementation Details

* **Augmentation:** Applied random flips, scaling, and color jitter to improve model robustness.
* **Transfer Learning:** Utilized pre-trained weights to accelerate convergence on custom safety classes.
* **Optimization:** Used `opencv-python-headless` and `.dockerignore` to minimize the container footprint.
