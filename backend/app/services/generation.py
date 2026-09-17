import logging
import os

import ollama

from app.core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTIONS = (
    "You are answering questions about the paper \"Attention Is All You Need\" using "
    "only the provided context. Cite which source(s) you used by their [id] (e.g. "
    "\"[Table 2]\", \"[Figure 4]\", \"[3.2.2]\"). If the context doesn't contain the "
    "answer, say so plainly instead of guessing. Answer entirely in your own words: "
    "summarize and synthesize, never copy or repeat any sentence from the context "
    "verbatim. Keep the answer focused and no longer than needed. "
    "If the question is not actually about this paper, do not try to answer it from "
    "the context anyway -- say that it is outside what you can answer from the paper."
)

ATTNVIZ_CAPTIONS = {
    "Figure 3": "An example of the attention mechanism following long-distance "
                "dependencies in the encoder self-attention in layer 5 of 6, for the "
                "word 'making' completing the phrase 'making...more difficult'.",
    "Figure 4": "Two attention heads, also in layer 5 of 6, apparently involved in "
                "anaphora resolution for the word 'its'.",
    "Figure 5": "Two attention heads exhibiting behaviour related to sentence "
                "structure, from the encoder self-attention at layer 5 of 6.",
}


def _ground_figure_with_vlm(chunk: dict, question: str) -> str:
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

    resp = ollama.chat(model=settings.vlm_model, messages=[{"role": "user", "content": prompt, "images": [image_path]}])
    logger.info("VLM grounding for %s: %d output tokens", chunk["id"], resp.eval_count or 0)
    return resp.message.content.strip()


def _build_prompt(chunks: list[dict], question: str) -> str:
    blocks = []
    for chunk in chunks:
        text = chunk["text"]
        if chunk["metadata"]["modality"] in ("figure_diagram", "figure_attnviz"):
            grounding = _ground_figure_with_vlm(chunk, question)
            text = text + " Additional detail grounded in the actual image: " + grounding
        meta = chunk["metadata"]
        blocks.append(f"[{chunk['id']}] ({meta['modality']}, page {meta['page']})\n{text}")
    context = "\n\n".join(blocks)
    return f"{SYSTEM_INSTRUCTIONS}\n\nContext:\n{context}\n\nQuestion: {question}"


def generate_answer(chunks: list[dict], question: str) -> dict:
    """Builds the grounded prompt (with query-time VLM grounding for any figure
    chunks) and calls the main LLM. Returns answer text + source ids + image refs."""
    prompt = _build_prompt(chunks, question)
    resp = ollama.chat(model=settings.llm_model, messages=[{"role": "user", "content": prompt}])
    logger.info(
        "generation done: prompt_tokens=%s output_tokens=%s",
        resp.prompt_eval_count, resp.eval_count,
    )

    images = []
    for chunk in chunks:
        image_path = chunk["metadata"].get("image_path")
        if image_path:
            filename = os.path.basename(image_path)
            images.append({
                "chunk_id": chunk["id"],
                "caption": chunk["text"].split("\n\n")[0][:200],
                "url": f"/static/images/{filename}",
            })

    return {
        "answer": resp.message.content.strip(),
        "sources": [c["id"] for c in chunks],
        "images": images,
    }
