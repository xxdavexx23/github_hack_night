import weaviate
from weaviate.classes.data import DataObject
from weaviate.classes.init import AdditionalConfig
from weaviate.classes.config import DataType

def push_to_weaviate(df):

    # ✅ Connect to Weaviate v4
    client = weaviate.connect_to_local(
        port=8080,
        grpc_port=50051,
        additional_config=AdditionalConfig(timeout=(10, 60))
    )

    df = df.collect()

    # Convert to dicts/lists
    data = df.to_pydict()
    paths = data["path"]
    embeddings = data["embedding"]
    descriptions = data["description"]


    # ✅ Iterate and upload each image entry
    for path_uri, embedding, description in zip(paths, embeddings, descriptions):
        path = path_uri.replace("file://", "")  # Strip URI prefix

        collection = client.collections.get("Photo")

        collection.data.insert(
            properties={
                "path": path,
                "description": description,
            },
            vector=embedding
        )


        try:
            print(f"✅ Uploaded: {os.path.basename(path)} – {description}")
        except Exception as e:
            print(f"❌ Failed to upload {path}: {e}")

    # ✅ Close client
    client.close()