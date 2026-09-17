import fitz, re, json

PDF = "Attention_is_all_you_need_paper.pdf"
doc = fitz.open(PDF)

TOP_NAMES = {"Abstract", "Conclusion", "Acknowledgements"}
STOP_AT = {"References"}
RUN_IN_LABELS = {"Encoder:", "Decoder:", "Residual Dropout", "Label Smoothing"}
NUM_RE = re.compile(r"^[1-9](\.[1-9]){0,2}$")

# hand-verified table extents (0-indexed page -> excluded y-range), from direct block-bbox inspection
TABLE_REGIONS = {
    5: (65, 190),   # Table 1
    7: (65, 246),   # Table 2
    8: (65, 390),   # Table 3
    9: (65, 241),   # Table 4
}

sections, current, pending_number = [], None, None
unrecognized_medi = []

def open_section(sec_id, title, page):
    global current
    current = {"id": sec_id, "title": title, "page_start": page + 1, "body": []}
    sections.append(current)

stop_walk = False
for pno in range(0, 10):
    if stop_walk:
        break
    page = doc[pno]
    d = page.get_text("dict")
    excl = TABLE_REGIONS.get(pno)
    for block in sorted(d["blocks"], key=lambda b: b["bbox"][1]):
        if stop_walk or "lines" not in block:
            continue
        for line in block["lines"]:
            spans = line["spans"]
            if not spans:
                continue
            text = "".join(s["text"] for s in spans).strip()
            if not text:
                continue

            y0 = line["bbox"][1]
            if y0 > 735:
                continue  # page-footer zone (page number)
            if excl and excl[0] <= y0 <= excl[1]:
                continue  # inside the known table region for this page -> skip entirely

            all_medi = all("Medi" in s["font"] for s in spans)

            if all_medi:
                if text in STOP_AT:
                    stop_walk = True
                    break
                if pending_number:
                    open_section(pending_number, text, pno)
                    pending_number = None
                    continue
                if NUM_RE.match(text):
                    pending_number = text
                    continue
                if text in TOP_NAMES:
                    open_section(text, text, pno)
                    continue
                if text in RUN_IN_LABELS:
                    if current is not None:
                        current["body"].append(text)
                    continue
                unrecognized_medi.append((pno + 1, text))
                if current is not None:
                    current["body"].append(text)
                continue

            if current is not None:
                current["body"].append(text)

chunks = []
for s in sections:
    body_text = re.sub(r"\s+", " ", " ".join(s["body"])).strip()
    word_count = len(body_text.split())
    label = f"{s['id']} {s['title']}" if not s['id'][0].isalpha() else s['title']
    chunks.append({
        "id": s["id"], "title": s["title"], "page_start": s["page_start"],
        "word_count": word_count, "text": f"{label}. {body_text}"
    })

# drop header-only chunks with no body text of their own (e.g. "6 Results",
# which is immediately followed by "6.1" with nothing in between)
dropped = [c["id"] for c in chunks if c["word_count"] == 0]
chunks = [c for c in chunks if c["word_count"] > 0]

with open("/home/claude/scratch/prose_chunks.json", "w") as f:
    json.dump(chunks, f, indent=2)

print(f"Total sections detected: {len(chunks)}\n")
for c in chunks:
    flag = "  <-- SUSPICIOUSLY SHORT" if c["word_count"] < 8 else ""
    print(f"[{c['id']:>10}] p{c['page_start']:>2} | {c['word_count']:>4} words | {c['title']}{flag}")

print(f"\nUnrecognized medi-font lines (expect only the title on p1): {unrecognized_medi}")
print(f"Dropped empty header-only chunks: {dropped}")
print(f"\nTotal words across all chunks: {sum(c['word_count'] for c in chunks)}")
