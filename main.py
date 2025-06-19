from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil
import uuid

app = FastAPI()

# Serve static files (uploaded images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Ensure static directory exists
os.makedirs("static", exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def root():
    # List all files in the static directory
    image_files = os.listdir("static")
    image_tags = ""
    
    for file in image_files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_tags += f"<img src='/static/{file}' width='300'><br>"

    return f"""
    <h1>Upload Images</h1>
    <form action="/upload" enctype="multipart/form-data" method="post">
        <input name="files" type="file" multiple>
        <button type="submit">Upload</button>
    </form>
    <h2>Uploaded Images</h2>
    {image_tags}
    """


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    saved_paths = []
    for file in files:
        filename = f"{uuid.uuid4()}_{file.filename}"
        path = os.path.join("static", filename)
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_paths.append(f"/static/{filename}")
    
    return {"uploaded_files": saved_paths}
