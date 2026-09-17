import fitz, os, json

PDF = "Attention_is_all_you_need_paper.pdf"
IMG_DIR = "data/processed/images"
os.makedirs(IMG_DIR, exist_ok=True)
doc = fitz.open(PDF)
ZOOM = 3  # ~216 dpi, legible for a vision-language model without huge file sizes


def render(pno, rect, out_path):
    page = doc[pno]
    pix = page.get_pixmap(clip=fitz.Rect(*rect), matrix=fitz.Matrix(ZOOM, ZOOM))
    pix.save(out_path)
    return out_path


def extract_caption_block(pno, label):
    """Find the text block that starts with 'Figure N:' and return its full text
    (multi-line captions are grouped into one PyMuPDF block)."""
    page = doc[pno]
    d = page.get_text("dict")
    for block in d["blocks"]:
        if "lines" not in block:
            continue
        block_text = " ".join(
            "".join(s["text"] for s in line["spans"]) for line in block["lines"]
        ).strip()
        if block_text.startswith(label):
            return " ".join(block_text.split())
    raise ValueError(f"caption block starting with {label!r} not found on page {pno + 1}")


# --- architecture diagrams: VLM description NOT generated yet (step 2 of the plan --
# pending explicit go-ahead to run gemma3:4b). Image + caption extracted now; "text"
# field holds the caption only until the VLM structured description is added. ---
DIAGRAM_FIGURES = [
    {"id": "Figure 1", "page_idx": 2, "rect": (190, 68, 422, 400), "image": "figure1.png"},
    {"id": "Figure 2", "page_idx": 3, "rect": (140, 65, 474, 271), "image": "figure2.png"},
]

# --- attention-viz figures: caption IS the primary retrievable content (per design,
# VLM only invoked on these at query time, not at ingestion). ---
ATTNVIZ_FIGURES = [
    {"id": "Figure 3", "page_idx": 12, "rect": (110, 82, 508, 305), "image": "figure3.png"},
    {"id": "Figure 4", "page_idx": 13, "rect": (115, 118, 508, 610), "image": "figure4.png"},
    {"id": "Figure 5", "page_idx": 14, "rect": (115, 133, 504, 599), "image": "figure5.png"},
]

chunks = []

for fig in DIAGRAM_FIGURES:
    image_path = f"{IMG_DIR}/{fig['image']}"
    render(fig["page_idx"], fig["rect"], image_path)
    caption = extract_caption_block(fig["page_idx"], fig["id"])
    chunks.append({
        "id": fig["id"],
        "title": caption.split(":", 1)[0],
        "page_start": fig["page_idx"] + 1,
        "modality": "figure_diagram",
        "word_count": len(caption.split()),
        "text": caption,
        "image_path": image_path,
        "vlm_description_pending": True,
    })

for fig in ATTNVIZ_FIGURES:
    image_path = f"{IMG_DIR}/{fig['image']}"
    render(fig["page_idx"], fig["rect"], image_path)
    caption = extract_caption_block(fig["page_idx"], fig["id"])
    chunks.append({
        "id": fig["id"],
        "title": caption.split(":", 1)[0],
        "page_start": fig["page_idx"] + 1,
        "modality": "figure_attnviz",
        "word_count": len(caption.split()),
        "text": caption,
        "image_path": image_path,
    })

with open("data/processed/figure_chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

print(f"Wrote {len(chunks)} figure chunks to data/processed/figure_chunks.json")
for c in chunks:
    pending = " (VLM description pending)" if c.get("vlm_description_pending") else ""
    print(f"[{c['id']:>8}] p{c['page_start']:>2} | {c['modality']:<16} | {c['image_path']}{pending}")
