import daft
from transformers import CLIPProcessor, CLIPModel
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os
from daft.datatype import DataType

# ==== Load Models ====
# CLIP for embeddings
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# BLIP for captions
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# ==== UDFs ====

# Generate CLIP embedding
def embed_image(uri: str) -> list[float]:
    path = uri.replace("file://", "")
    image = Image.open(path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)
    return embedding.squeeze().tolist()

# Generate BLIP caption
def describe_image(uri: str) -> str:
    path = uri.replace("file://", "")
    image = Image.open(path).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = blip_model.generate(**inputs)
    return blip_processor.decode(output[0], skip_special_tokens=True)

# ==== Load + Process ====

# Load image paths
df_jpg = daft.from_glob_path("static/*.jpeg")
df_png = daft.from_glob_path("static/*.png")
df = df_jpg.concat(df_png)


# Apply embedding
df = df.with_column(
    "embedding",
    df["path"].apply(embed_image, return_dtype=DataType.list(DataType.float32()))
)

# Apply description
df = df.with_column(
    "description",
    df["path"].apply(describe_image, return_dtype=DataType.string())
)

# ==== Output ====
df.show()
