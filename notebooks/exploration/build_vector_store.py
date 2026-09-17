import chromadb, json

with open("data/processed/all_chunks_embedded.json", encoding="utf-8") as f:
    chunks = json.load(f)

client = chromadb.PersistentClient(path="data/vector_store")
# drop and recreate so re-running this script after re-embedding doesn't leave stale entries
try:
    client.delete_collection("attention_paper")
except Exception:
    pass
collection = client.create_collection(
    name="attention_paper",
    metadata={"hnsw:space": "cosine"},
)

collection.add(
    ids=[c["id"] for c in chunks],
    embeddings=[c["embedding"] for c in chunks],
    documents=[c["text"] for c in chunks],
    metadatas=[
        {k: v for k, v in c["metadata"].items() if v is not None}
        for c in chunks
    ],
)

print(f"Persisted {collection.count()} chunks to data/vector_store/ (collection 'attention_paper')")
