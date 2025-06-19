import weaviate
from weaviate.classes.config import Property, DataType
from weaviate.classes.init import AdditionalConfig

# Connect to local instance
client = weaviate.connect_to_local(
    port=8080,
    additional_config=AdditionalConfig(timeout=(10, 60))  # optional
)

# Delete "Photo" class if it already exists
if client.collections.exists("Photo"):
    client.collections.delete("Photo")

# Create "Photo" class
client.collections.create(
    name="Photo",
    properties=[
        Property(name="path", data_type=DataType.TEXT),
        Property(name="description", data_type=DataType.TEXT),
    ],
    vectorizer_config=weaviate.classes.config.Configure.Vectorizer.none()
)

print("✅ Schema created.")
client.close()