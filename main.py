from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import io
from PIL import Image

app = FastAPI()

# Load your fine-tuned model
model = YOLO("weights/best.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read the uploaded image
    request_object_content = await file.read()
    img = Image.open(io.BytesIO(request_object_content))

    # 2. Run inference
    results = model(img)

    # 3. Parse results into JSON format
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy.tolist()[0]  # [x1, y1, x2, y2]
            })

    return {"detections": detections}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)