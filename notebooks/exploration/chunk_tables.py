import fitz, re, json

PDF = "Attention_is_all_you_need_paper.pdf"
doc = fitz.open(PDF)

# same bboxes as chunk_prose.py TABLE_REGIONS -- reused here, not re-derived
TABLE_REGIONS = {
    5: (65, 190),   # Table 1
    7: (65, 246),   # Table 2
    8: (65, 390),   # Table 3
    9: (65, 241),   # Table 4
}


# Captions transcribed directly from the rendered PDF text (see inspect2.py output) --
# not worth generalizing a caption/header-row boundary detector for 4 known tables;
# this is a bespoke single-document pipeline, consistent with TABLE_REGIONS above.
CAPTIONS = {
    5: "Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential "
       "operations for different layer types. n is the sequence length, d is the representation "
       "dimension, k is the kernel size of convolutions and r the size of the neighborhood in "
       "restricted self-attention.",
    7: "Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art "
       "models on the English-to-German and English-to-French newstest2014 tests at a fraction "
       "of the training cost.",
    8: "Table 3: Variations on the Transformer architecture. Unlisted values are identical to "
       "those of the base model. All metrics are on the English-to-German translation "
       "development set, newstest2013. Listed perplexities are per-wordpiece, according to our "
       "byte-pair encoding, and should not be compared to per-word perplexities.",
    9: "Table 4: The Transformer generalizes well to English constituency parsing (Results are "
       "on Section 23 of WSJ).",
}

# lead-in sentence pulled from the prose chunk that first discusses each table
# (found via regex search over prose_chunks.json -- see find_table_refs.py)
CONTEXT = {
    5: "As noted in Table 1, a self-attention layer connects all positions with a constant "
       "number of sequentially executed operations, whereas a recurrent layer requires O(n) "
       "sequential operations.",
    7: "Table 2 summarizes our results and compares our translation quality and training costs "
       "to other model architectures from the literature.",
    8: "We present these results in Table 3.",
    9: "Our results in Table 4 show that despite the lack of task-specific tuning our model "
       "performs surprisingly well, yielding better results than all previously reported "
       "models with the exception of the Recurrent Neural Network Grammar [8].",
}

# Hand-transcribed cell content, cross-checked line-by-line against the raw (x, y, text)
# span dump for each TABLE_REGIONS bbox. page.find_tables() only recovers Table 3 and 4
# (and mangles Table 3's merged row-groups), and misses Table 1/2 entirely, so -- consistent
# with this project's existing "bespoke, not generalized" approach to this one document --
# these are transcribed once and hardcoded rather than re-parsed on every run.

TABLE_1_MD = """| Layer Type | Complexity per Layer | Sequential Operations | Maximum Path Length |
|---|---|---|---|
| Self-Attention | O(n^2 * d) | O(1) | O(1) |
| Recurrent | O(n * d^2) | O(n) | O(n) |
| Convolutional | O(k * n * d^2) | O(1) | O(log_k(n)) |
| Self-Attention (restricted) | O(r * n * d) | O(1) | O(n/r) |"""

TABLE_2_MD = """| Model | BLEU EN-DE | BLEU EN-FR | Training Cost (FLOPs) EN-DE | Training Cost (FLOPs) EN-FR |
|---|---|---|---|---|
| ByteNet | 23.75 | - | - | - |
| Deep-Att + PosUnk | - | 39.2 | - | 1.0e20 |
| GNMT + RL | 24.6 | 39.92 | 2.3e19 | 1.4e20 |
| ConvS2S | 25.16 | 40.46 | 9.6e18 | 1.5e20 |
| MoE | 26.03 | 40.56 | 2.0e19 | 1.2e20 |
| Deep-Att + PosUnk Ensemble | - | 40.4 | - | 8.0e20 |
| GNMT + RL Ensemble | 26.30 | 41.16 | 1.8e20 | 1.1e21 |
| ConvS2S Ensemble | 26.36 | 41.29 | 7.7e19 | 1.2e21 |
| Transformer (base model) | 27.3 | 38.1 | 3.3e18 (combined) | 3.3e18 (combined) |
| Transformer (big) | 28.4 | 41.8 | 2.3e19 (combined) | 2.3e19 (combined) |"""

TABLE_3_MD = """| Row | N | d_model | d_ff | h | d_k | d_v | P_drop | eps_ls | train steps | PPL (dev) | BLEU (dev) | params (x10^6) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 6 | 512 | 2048 | 8 | 64 | 64 | 0.1 | 0.1 | 100K | 4.92 | 25.8 | 65 |
| (A) | - | - | - | 1 | 512 | 512 | - | - | - | 5.29 | 24.9 | - |
| (A) | - | - | - | 4 | 128 | 128 | - | - | - | 5.00 | 25.5 | - |
| (A) | - | - | - | 16 | 32 | 32 | - | - | - | 4.91 | 25.8 | - |
| (A) | - | - | - | 32 | 16 | 16 | - | - | - | 5.01 | 25.4 | - |
| (B) | - | - | - | - | 16 | - | - | - | - | 5.16 | 25.1 | 58 |
| (B) | - | - | - | - | 32 | - | - | - | - | 5.01 | 25.4 | 60 |
| (C) | 2 | - | - | - | - | - | - | - | - | 6.11 | 23.7 | 36 |
| (C) | 4 | - | - | - | - | - | - | - | - | 5.19 | 25.3 | 50 |
| (C) | 8 | - | - | - | - | - | - | - | - | 4.88 | 25.5 | 80 |
| (C) | - | 256 | - | - | 32 | 32 | - | - | - | 5.75 | 24.5 | 28 |
| (C) | - | 1024 | - | - | 128 | 128 | - | - | - | 4.66 | 26.0 | 168 |
| (C) | - | - | 1024 | - | - | - | - | - | - | 5.12 | 25.4 | 53 |
| (C) | - | - | 4096 | - | - | - | - | - | - | 4.75 | 26.2 | 90 |
| (D) | - | - | - | - | - | - | 0.0 | - | - | 5.77 | 24.6 | - |
| (D) | - | - | - | - | - | - | 0.2 | - | - | 4.95 | 25.5 | - |
| (D) | - | - | - | - | - | - | - | 0.0 | - | 4.67 | 25.3 | - |
| (D) | - | - | - | - | - | - | - | 0.2 | - | 5.47 | 25.7 | - |
| (E) | - | - | - | - | - | - | - | - | - | 4.92 | 25.7 | - |
| big | 6 | 1024 | 4096 | 16 | - | - | 0.3 | - | 300K | 4.33 | 26.4 | 213 |

Row-group notes: (A) varies number of attention heads h and the attention key/value
dimensions d_k, d_v while keeping total computation constant. (B) varies only d_k. (C)
varies model size (number of layers N, then d_model/d_ff together). (D) varies
regularization (dropout P_drop, then label smoothing eps_ls). (E) replaces sinusoidal
positional encoding with a learned positional embedding. Unlisted cells in a row are
identical to the base model's values, per the paper's own caption."""

TABLE_4_MD = """| Parser | Training | WSJ 23 F1 |
|---|---|---|
| Vinyals & Kaiser et al. (2014) | WSJ only, discriminative | 88.3 |
| Petrov et al. (2006) | WSJ only, discriminative | 90.4 |
| Zhu et al. (2013) | WSJ only, discriminative | 90.4 |
| Dyer et al. (2016) | WSJ only, discriminative | 91.7 |
| Transformer (4 layers) | WSJ only, discriminative | 91.3 |
| Zhu et al. (2013) | semi-supervised | 91.3 |
| Huang & Harper (2009) | semi-supervised | 91.3 |
| McClosky et al. (2006) | semi-supervised | 92.1 |
| Vinyals & Kaiser et al. (2014) | semi-supervised | 92.1 |
| Transformer (4 layers) | semi-supervised | 92.7 |
| Luong et al. (2015) | multi-task | 93.0 |
| Dyer et al. (2016) | generative | 93.3 |"""

TABLES_MD = {5: TABLE_1_MD, 7: TABLE_2_MD, 8: TABLE_3_MD, 9: TABLE_4_MD}
TABLE_IDS = {5: "Table 1", 7: "Table 2", 8: "Table 3", 9: "Table 4"}

chunks = []
for pno in sorted(TABLE_REGIONS):
    table_id = TABLE_IDS[pno]
    caption = CAPTIONS[pno]
    context = CONTEXT[pno]
    md = TABLES_MD[pno]
    text = f"{caption}\n\nContext: {context}\n\n{md}"
    chunks.append({
        "id": table_id,
        "title": caption.split(":", 1)[0] + ":" + caption.split(":", 1)[1].split(".")[0],
        "page_start": pno + 1,
        "modality": "table",
        "word_count": len(text.split()),
        "text": text,
        "image_path": None,
    })

with open("data/processed/table_chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

print(f"Wrote {len(chunks)} table chunks to data/processed/table_chunks.json")
for c in chunks:
    print(f"[{c['id']:>8}] p{c['page_start']:>2} | {c['word_count']:>4} words | {c['title']}")
