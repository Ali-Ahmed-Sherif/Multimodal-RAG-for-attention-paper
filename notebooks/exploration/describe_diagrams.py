import ollama, json, sys, re

PROMPT = """This image is an architecture diagram from the paper "Attention Is All You Need".
Describe it structurally for someone who cannot see the image, so the description can be
used to answer questions about the architecture. Cover:
1. What are the distinct components/blocks (name them).
2. How are they connected -- what feeds into what, in what order (top-to-bottom or
   left-to-right flow).
3. Any repeated/stacked structures (e.g. "Nx" meaning the block repeats N times).
4. Any labeled inputs/outputs.
Be precise about connections (arrows), not just a list of boxes. Do not offer further help
or ask a follow-up question at the end -- end after the description itself."""

FIGURES = {
    "Figure 1": "data/processed/images/figure1.png",
    "Figure 2": "data/processed/images/figure2.png",
}

target = sys.argv[1] if len(sys.argv) > 1 else None

with open("data/processed/figure_chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

for name, path in FIGURES.items():
    if target and name != target:
        continue
    print(f"\n=== Describing {name} ({path}) ===")
    resp = ollama.chat(
        model="gemma3:4b",
        messages=[{"role": "user", "content": PROMPT, "images": [path]}],
    )
    description = resp["message"]["content"].strip()
    # strip a trailing conversational offer-to-help line if the model added one anyway
    description = re.sub(r"\n+---\n+Would you like.*$", "", description, flags=re.DOTALL).strip()
    print(description)

    for c in chunks:
        if c["id"] == name:
            caption = c["text"].split("\n\nStructural description")[0]  # in case of re-run
            c["text"] = caption + "\n\nStructural description (gemma3:4b): " + description
            c["word_count"] = len(c["text"].split())
            c["vlm_description_pending"] = False

with open("data/processed/figure_chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

print("\nfigure_chunks.json updated.")
