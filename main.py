from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import shutil
import uuid
import hdbscan
import numpy as np
from sklearn.metrics.pairwise import cosine_distances


app = FastAPI()

# Serve static files (uploaded images)
app.mount("/static", StaticFiles(directory="static"), name="static")
os.makedirs("static", exist_ok=True)

# Load CLIP model and processor
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <h1>Upload Images</h1>
    <form action="/upload" enctype="multipart/form-data" method="post">
        <input name="files" type="file" multiple>
        <button type="submit">Upload</button>
    </form>
    """

@app.post("/upload", response_class=HTMLResponse)
async def upload(files: list[UploadFile] = File(...)):
    saved_paths = []
    images = []

    for file in files:
        filename = f"{uuid.uuid4()}_{file.filename}"
        path = os.path.join("static", filename)
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_paths.append(f"/static/{filename}")
        images.append(Image.open(path).convert('RGB'))

    # Generate CLIP embeddings
    inputs = processor(images=images, return_tensors="pt", padding=True)
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs).numpy().astype(np.float64)

    distance_matrix = cosine_distances(embeddings).astype(np.float64)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,          # minimum number of images per cluster
        min_samples=1,               # encourages tighter clusters and fewer noise points
        metric='precomputed',
        cluster_selection_method='eom',
        allow_single_cluster=False,  # avoids having all images in one big cluster
        cluster_selection_epsilon=0.05  # adjusts sensitivity of clustering; small values make tighter clusters
    )

    labels = clusterer.fit_predict(distance_matrix)


    # Organize images by clusters
    clustered_images = {}
    for idx, label in enumerate(labels):
        clustered_images.setdefault(label, []).append(saved_paths[idx])

    # Render results in HTML
    html_content = "<h2>Image Clusters</h2>"
    for label, paths in clustered_images.items():
        html_content += f"<h3>Cluster {label if label != -1 else 'Noise'}</h3><div>"
        for img_path in paths:
            html_content += f'<img src="{img_path}" width="150" style="margin:5px;">'
        html_content += "</div><hr>"

    return html_content
