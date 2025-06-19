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
    return """
    <h1>Upload Images</h1>
    <form action="/upload" enctype="multipart/form-data" method="post">
        <input name="files" type="file" multiple>
        <button type="submit">Upload</button>
    </form>
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
