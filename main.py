import os
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model setup
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.eval()

class_idx = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]

# Image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def process_image(image):
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    _, predicted_idx = torch.max(output, 1)
    confidence = probabilities[predicted_idx].item()
    return class_idx[predicted_idx.item()], confidence

@app.get("/")
async def read_index():
    return FileResponse('index.html')

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        class_name, confidence = process_image(image)
        return JSONResponse(content={"detected_object": class_name, "confidence": f"{confidence:.2%}"})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.post("/api/classify")
async def classify_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        class_name, confidence = process_image(image)
        return JSONResponse(content={
            "class_label": class_name,
            "confidence": f"{confidence:.2%}"
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
