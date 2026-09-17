import ollama, json

# nomic-embed-text was trained with asymmetric task prefixes: corpus/document text gets
# "search_document: ", queries get "search_query: " at retrieval time (not applied here --
# this script only embeds the corpus side). Omitting this prefix measurably hurts
# retrieval quality for this model; keep it in sync with whatever prefixes the query-side
# embedding call in the backend/notebook.
DOCUMENT_PREFIX = "search_document: "

with open("data/processed/all_chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [DOCUMENT_PREFIX + c["text"] for c in chunks]
resp = ollama.embed(model="nomic-embed-text", input=texts)
embeddings = resp["embeddings"]

assert len(embeddings) == len(chunks)
for c, emb in zip(chunks, embeddings):
    c["embedding"] = emb

with open("data/processed/all_chunks_embedded.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f)

dims = {len(e) for e in embeddings}
print(f"Embedded {len(chunks)} chunks with nomic-embed-text, dims={dims}")
print("Wrote data/processed/all_chunks_embedded.json")
