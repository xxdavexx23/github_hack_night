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
from fastapi import Form
from weaviate_helper.connect import connect_client  # Create this helper
import numpy as np

app = FastAPI()

# Serve static files (uploaded images)
app.mount("/static", StaticFiles(directory="static"), name="static")
os.makedirs("static", exist_ok=True)

# Load CLIP model and processor
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)


@app.get("/search", response_class=HTMLResponse)
async def search_page():
    return """
    <h1>Search Images</h1>
    <form action="/search" method="post">
        <input name="query" type="text">
        <button type="submit">Search</button>
    </form>
    """

@app.post("/search", response_class=HTMLResponse)
async def search_images(query: str = Form(...)):
    # 1. Convert query to embedding
    inputs = processor(text=[query], return_tensors="pt", padding=True)
    with torch.no_grad():
        query_embedding = model.get_text_features(**inputs).numpy().astype(np.float32).flatten()

    # 2. Connect to Weaviate
    client = connect_client()

    # 3. Perform vector search
    photo_collection = client.collections.get("Photo")

    results = photo_collection.query.near_vector(
        near_vector=query_embedding.tolist(),
        limit=10
    )

    # 4. Render results
    if not results.objects:
        return "<h2>No matches found.</h2>"

    html_content = "<h2>Search Results</h2><div>"
    for obj in results.objects:
        description = obj.properties.get("description", "")
        path = obj.properties.get("path", "")
        html_content += f'<p>{description}</p>'
        html_content += f'<img src="{path}" width="150" style="margin:5px;">'
    html_content += "</div>"

    return html_content

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


    from utils.daft_descriptor import describe_image, create_dataframe, apply

    df = create_dataframe()

    df = apply(df)

    from weaviate_helper.push_to_weaviate import push_to_weaviate

    push_to_weaviate(df)

    
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
