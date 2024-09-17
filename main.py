import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import io
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()

# Serve index.html at the root
@app.get("/")
async def read_index():
    return FileResponse('index.html')

# Load a pre-trained ResNet50 model
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.eval()

# Get the class index
class_idx = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    # Read and transform the image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')  # Convert to RGB
    image_tensor = transform(image).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)

    # Get the predicted class
    _, predicted_idx = torch.max(output, 1)
    predicted_label = predicted_idx.item()

    # Get the class name
    class_name = class_idx[predicted_label]

    return JSONResponse(content={"detected_object": class_name})

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    return PlainTextResponse(str(exc.detail), status_code=exc.status_code)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=400)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Test function for local image detection
def test_local_image_detection(image_path):
    # Open and transform the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)

    # Get the predicted class
    _, predicted_idx = torch.max(output, 1)
    predicted_label = predicted_idx.item()

    # Get the class name
    class_name = class_idx[predicted_label]

    print(f"Detected object in {image_path}: {class_name}")

if __name__ == "__main__":
    # Test local image detection
    # test_image_path = "/Desktop/tmp/tmpimages/clock.jpg"  # Replace with the path to your test image
    # if os.path.exists(test_image_path):
    #     test_local_image_detection(test_image_path)
    # else:
    #     print(f"Test image not found at {test_image_path}")

    # Run the FastAPI server
    import uvicorn
    port = int(os.environ.get("PORT", 8009))
    uvicorn.run(app, host="0.0.0.0", port=port)
