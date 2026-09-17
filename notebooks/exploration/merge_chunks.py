import json

with open("data/processed/prose_chunks.json", encoding="utf-8") as f:
    prose = json.load(f)
with open("data/processed/table_chunks.json", encoding="utf-8") as f:
    tables = json.load(f)
with open("data/processed/figure_chunks.json", encoding="utf-8") as f:
    figures = json.load(f)

unified = []

for c in prose:
    unified.append({
        "id": c["id"],
        "text": c["text"],
        "metadata": {
            "modality": "text",
            "section": c["title"],
            "page": c["page_start"],
            "image_path": None,
        },
    })

for c in tables:
    unified.append({
        "id": c["id"],
        "text": c["text"],
        "metadata": {
            "modality": "table",
            "section": c["id"],
            "page": c["page_start"],
            "image_path": None,
        },
    })

for c in figures:
    unified.append({
        "id": c["id"],
        "text": c["text"],
        "metadata": {
            "modality": c["modality"],
            "section": c["id"],
            "page": c["page_start"],
            "image_path": c["image_path"],
        },
    })

ids = [c["id"] for c in unified]
assert len(ids) == len(set(ids)), f"duplicate ids: {[i for i in ids if ids.count(i) > 1]}"

with open("data/processed/all_chunks.json", "w", encoding="utf-8") as f:
    json.dump(unified, f, indent=2)

by_modality = {}
for c in unified:
    by_modality[c["metadata"]["modality"]] = by_modality.get(c["metadata"]["modality"], 0) + 1

print(f"Wrote {len(unified)} unified chunks to data/processed/all_chunks.json")
for modality, count in sorted(by_modality.items()):
    print(f"  {modality}: {count}")
