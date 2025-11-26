from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uuid

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

client = QdrantClient(
    url="https://0f4c45ec-a80a-4979-ac46-48044746ee13.europe-west3-0.gcp.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.2YY8qnJiiCJ7G4n2r9ryHRJa5RlzziLuExfHJSeIh2Q"
)

collection_name = "erpnext_items"

if collection_name not in [c.name for c in client.get_collections().collections]:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

app = FastAPI()

class Item(BaseModel):
    item_name: str

class BestSearchQuery(BaseModel):
    query: str

def get_embedding(text: str):
    return model.encode(text).tolist()

@app.post("/add-item")
def add_item(item: Item):
    vector = get_embedding(item.item_name)

    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload={"name": item.item_name}
    )

    client.upsert(
        collection_name=collection_name,
        points=[point]
    )

    return {"message": "stored", "vector_dimension": len(vector)}

@app.get("/all")
def list_all():
    points, _ = client.scroll(
        collection_name=collection_name,
        limit=1000,
        with_payload=True,
        with_vectors=False
    )
    return points

@app.post("/semantic_search")
def semantic_search(data: BestSearchQuery):
    query_vector = get_embedding(data.query)

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=1,
        with_payload=True,
        with_vectors=False
    )

    if not results or not results.points:
        return {"results": []}

    hit = results.points[0]

    return {
        "results": [
            {
                "name": hit.payload.get("name"),
                "score": hit.score
            }
        ]
    }
