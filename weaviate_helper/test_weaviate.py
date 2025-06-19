import weaviate
from weaviate.classes.config import Property, DataType
from weaviate.classes.init import AdditionalConfig

# Connect to local instance
client = weaviate.connect_to_local(
    port=8080,
    additional_config=AdditionalConfig(timeout=(10, 60))  # optional
)

collection = client.collections.get("Photo")
results = collection.query.fetch_objects(limit=5)

for obj in results.objects:
    print(obj.properties)

client.close()  