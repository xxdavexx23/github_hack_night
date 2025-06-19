import weaviate
from weaviate.classes.init import AdditionalConfig

def connect_client():
    return weaviate.connect_to_local(
        port=8080,
        grpc_port=50051,
        additional_config=AdditionalConfig(timeout=(10, 60))
    )
