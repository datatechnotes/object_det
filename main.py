import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import io

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
    _, predicted_idx = torch.max(output, 1)
    return class_idx[predicted_idx.item()]

@app.get("/")
async def read_index():
    return FileResponse('index.html')

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    class_name = process_image(image)
    return JSONResponse(content={"detected_object": class_name})

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(content={"error": str(exc.detail)}, status_code=exc.status_code)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(content={"error": str(exc)}, status_code=400)

def test_local_image_detection(image_path):
    image = Image.open(image_path).convert('RGB')
    class_name = process_image(image)
    print(f"Detected object in {image_path}: {class_name}")

if __name__ == "__main__":
    # Test local image detection
    # test_image_path = "/Desktop/tmp/tmpimages/clock.jpg"  # Replace with the path to your test image
    # if os.path.exists(test_image_path):
    #     test_local_image_detection(test_image_path)
    # else:
    #     print(f"Test image not found at {test_image_path}")

    import uvicorn
    port = int(os.environ.get("PORT", 8009))
    uvicorn.run(app, host="0.0.0.0", port=port)
