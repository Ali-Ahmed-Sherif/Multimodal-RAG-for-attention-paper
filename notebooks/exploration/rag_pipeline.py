import json, re, time
import numpy as np
import ollama
import chromadb
from rank_bm25 import BM25Okapi

EMBED_MODEL = "nomic-embed-text"
VLM_MODEL = "gemma3:4b"
LLM_MODEL = "llama3.2:1b"
QUERY_PREFIX = "search_query: "  # nomic-embed-text asymmetric prefix, query side
TOP_K = 4
RRF_K = 60  # standard reciprocal-rank-fusion constant

with open("data/processed/all_chunks_embedded.json", encoding="utf-8") as f:
    CHUNKS = json.load(f)
CHUNK_BY_ID = {c["id"]: c for c in CHUNKS}

_client = chromadb.PersistentClient(path="data/vector_store")
_collection = _client.get_collection("attention_paper")


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


_bm25_corpus_ids = [c["id"] for c in CHUNKS]
_bm25 = BM25Okapi([_tokenize(c["text"]) for c in CHUNKS])


def ns_to_s(ns):
    return round(ns / 1e9, 3) if ns is not None else None


def ollama_ps_snapshot():
    resp = ollama.ps()
    return [
        {
            "model": m.model,
            "size_vram_gb": round(m.size_vram / 1e9, 2),
            "context_length": m.context_length,
        }
        for m in resp.models
    ]


def embed_query(question):
    t0 = time.time()
    resp = ollama.embed(model=EMBED_MODEL, input=QUERY_PREFIX + question)
    vec = resp.embeddings[0]
    return {
        "vector": vec,
        "dims": len(vec),
        "l2_norm": round(float(np.linalg.norm(vec)), 4),
        "preview": [round(v, 4) for v in vec[:6]],
        "wall_time_s": round(time.time() - t0, 3),
        "ollama_metrics": {
            "total_duration_s": ns_to_s(resp.total_duration),
            "load_duration_s": ns_to_s(resp.load_duration),
        },
    }


def dense_rank_all(query_vector):
    """Full ranking of all chunks by cosine distance, via the persisted Chroma store."""
    res = _collection.query(query_embeddings=[query_vector], n_results=len(CHUNKS))
    return [
        {"id": id_, "distance": round(dist, 4)}
        for id_, dist in zip(res["ids"][0], res["distances"][0])
    ]


def bm25_rank_all(question):
    scores = _bm25.get_scores(_tokenize(question))
    ranked = sorted(zip(_bm25_corpus_ids, scores), key=lambda x: -x[1])
    return [{"id": id_, "score": round(float(s), 4)} for id_, s in ranked]


def hybrid_retrieve(question, query_vector, k=TOP_K):
    dense = dense_rank_all(query_vector)
    bm25 = bm25_rank_all(question)
    dense_rank = {d["id"]: i + 1 for i, d in enumerate(dense)}
    bm25_rank = {d["id"]: i + 1 for i, d in enumerate(bm25)}

    fused = []
    for cid in dense_rank:
        rd, rb = dense_rank[cid], bm25_rank[cid]
        fused.append({
            "id": cid,
            "dense_rank": rd,
            "bm25_rank": rb,
            "fused_score": round(1 / (RRF_K + rd) + 1 / (RRF_K + rb), 6),
        })
    fused.sort(key=lambda x: -x["fused_score"])
    top = fused[:k]
    for entry in top:
        chunk = CHUNK_BY_ID[entry["id"]]
        entry["modality"] = chunk["metadata"]["modality"]
        entry["page"] = chunk["metadata"]["page"]
        entry["preview"] = chunk["text"][:120].replace("\n", " ")
    return {"dense_full": dense, "bm25_full": bm25, "fused_all": fused, "top_k": top}


ATTNVIZ_CAPTIONS = {
    "Figure 3": "An example of the attention mechanism following long-distance "
                "dependencies in the encoder self-attention in layer 5 of 6, for the "
                "word 'making' completing the phrase 'making...more difficult'.",
    "Figure 4": "Two attention heads, also in layer 5 of 6, apparently involved in "
                "anaphora resolution for the word 'its'.",
    "Figure 5": "Two attention heads exhibiting behaviour related to sentence "
                "structure, from the encoder self-attention at layer 5 of 6.",
}


def ground_figure_with_vlm(chunk, question):
    image_path = chunk["metadata"]["image_path"]
    modality = chunk["metadata"]["modality"]

    if modality == "figure_attnviz":
        caption = ATTNVIZ_CAPTIONS.get(chunk["id"], "")
        prompt = (
            f"This image is from the paper 'Attention Is All You Need'. The paper's "
            f"own caption for it says: \"{caption}\" A user is asking: \"{question}\" "
            f"Using the image AND the caption above as your anchor (the caption is "
            f"verified-accurate; don't contradict it), answer the user's question in "
            f"2-4 sentences. If the question asks about fine details like which "
            f"specific line is thickest, defer to the caption's claim rather than "
            f"guessing from pixels."
        )
    else:
        prompt = (
            f"This image is an architecture diagram from the paper 'Attention Is All "
            f"You Need'. A user is asking: \"{question}\" Answer their question in "
            f"4-6 sentences using the diagram, describing only the relevant "
            f"components and connections you can actually see. Be concise."
        )

    t0 = time.time()
    resp = ollama.chat(model=VLM_MODEL, messages=[{"role": "user", "content": prompt, "images": [image_path]}])
    return {
        "chunk_id": chunk["id"],
        "image_path": image_path,
        "prompt": prompt,
        "response": resp.message.content.strip(),
        "wall_time_s": round(time.time() - t0, 3),
        "ollama_metrics": {
            "total_duration_s": ns_to_s(resp.total_duration),
            "load_duration_s": ns_to_s(resp.load_duration),
            "prompt_eval_count": resp.prompt_eval_count,
            "eval_count": resp.eval_count,
        },
    }


SYSTEM_INSTRUCTIONS = (
    "You are answering questions about the paper \"Attention Is All You Need\" using "
    "only the provided context. Cite which source(s) you used by their [id] (e.g. "
    "\"[Table 2]\", \"[Figure 4]\", \"[3.2.2]\"). If the context doesn't contain the "
    "answer, say so plainly instead of guessing. Answer entirely in your own words: "
    "summarize and synthesize, never copy or repeat any sentence from the context "
    "verbatim. Keep the answer focused and no longer than needed."
)


def build_prompt(top_k, vlm_groundings, question):
    grounding_by_id = {g["chunk_id"]: g["response"] for g in vlm_groundings}
    blocks = []
    for entry in top_k:
        chunk = CHUNK_BY_ID[entry["id"]]
        text = chunk["text"]
        if entry["id"] in grounding_by_id:
            text = text + " Additional detail grounded in the actual image: " + grounding_by_id[entry["id"]]
        blocks.append(f"[{entry['id']}] ({entry['modality']}, page {entry['page']})\n{text}")
    context = "\n\n".join(blocks)
    return f"{SYSTEM_INSTRUCTIONS}\n\nContext:\n{context}\n\nQuestion: {question}"


def generate_answer(prompt):
    t0 = time.time()
    resp = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    return {
        "answer": resp.message.content.strip(),
        "wall_time_s": round(time.time() - t0, 3),
        "ollama_metrics": {
            "total_duration_s": ns_to_s(resp.total_duration),
            "load_duration_s": ns_to_s(resp.load_duration),
            "prompt_eval_count": resp.prompt_eval_count,
            "prompt_eval_duration_s": ns_to_s(resp.prompt_eval_duration),
            "eval_count": resp.eval_count,
            "eval_duration_s": ns_to_s(resp.eval_duration),
        },
    }


def run_query(question, expected_source=None):
    t_start = time.time()
    ps_before = ollama_ps_snapshot()

    embed_trace = embed_query(question)
    retrieval_trace = hybrid_retrieve(question, embed_trace["vector"], k=TOP_K)
    top_ids = [e["id"] for e in retrieval_trace["top_k"]]
    hit = expected_source in top_ids if expected_source else None

    vlm_traces = []
    for entry in retrieval_trace["top_k"]:
        chunk = CHUNK_BY_ID[entry["id"]]
        if chunk["metadata"]["modality"] in ("figure_diagram", "figure_attnviz"):
            vlm_traces.append(ground_figure_with_vlm(chunk, question))

    prompt = build_prompt(retrieval_trace["top_k"], vlm_traces, question)
    generation_trace = generate_answer(prompt)

    ps_after = ollama_ps_snapshot()

    image_paths = [
        CHUNK_BY_ID[e["id"]]["metadata"]["image_path"]
        for e in retrieval_trace["top_k"]
        if CHUNK_BY_ID[e["id"]]["metadata"].get("image_path")
    ]

    return {
        "question": question,
        "expected_source": expected_source,
        "retrieval_hit": hit,
        "embed": {k: v for k, v in embed_trace.items() if k != "vector"},
        "dense_full_ranking": retrieval_trace["dense_full"],
        "bm25_full_ranking": retrieval_trace["bm25_full"],
        "fused_top_k": retrieval_trace["top_k"],
        "vlm_groundings": vlm_traces,
        "final_prompt": prompt,
        "final_prompt_chars": len(prompt),
        "generation": generation_trace,
        "answer": generation_trace["answer"],
        "image_paths_for_ui": image_paths,
        "ollama_ps_before": ps_before,
        "ollama_ps_after": ps_after,
        "total_wall_time_s": round(time.time() - t_start, 3),
    }
