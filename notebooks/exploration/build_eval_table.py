import json

with open("data/processed/eval_trace.json", encoding="utf-8") as f:
    trace = json.load(f)

lines = [
    "# Phase 2.6 — Evaluation Results",
    "",
    f"{len(trace)} test questions run end-to-end through the hybrid retrieval + "
    f"query-time VLM grounding + `llama3.2:1b` generation pipeline "
    f"(`notebooks/exploration/run_test_queries.py`). Full per-case trace: "
    f"`data/processed/eval_trace.json` / `eval_trace_console.txt`.",
    "",
    "| # | Question | Retrieved source(s) (top-4) | Relevant? | Answer (truncated) | Grounded? |",
    "|---|---|---|---|---|---|",
]

for i, r in enumerate(trace, 1):
    q = r["question"][:70] + ("..." if len(r["question"]) > 70 else "")
    sources = ", ".join(e["id"] for e in r["fused_top_k"])
    if r["expected_source"] is None:
        relevant = "n/a"
    else:
        relevant = "Yes" if r["retrieval_hit"] else "**No**"
    answer = r["answer"][:110].replace("\n", " ") + ("..." if len(r["answer"]) > 110 else "")
    grounded = "Yes" if r["retrieval_hit"] or r["expected_source"] is None else "**No (wrong source)**"
    lines.append(f"| {i} | {q} | {sources} | {relevant} | {answer} | {grounded} |")

hits = [r for r in trace if r["retrieval_hit"] is True]
scored = [r for r in trace if r["retrieval_hit"] is not None]

lines += [
    "",
    f"**Retrieval hit rate: {len(hits)}/{len(scored)} ({100*len(hits)/len(scored):.0f}%)** "
    f"against a single expected source id per question (one question has no single "
    f"expected source by design and is marked n/a above).",
    "",
    "## Main failure cases and mitigations",
    "",
    "1. **Wrong source retrieved -> confident hallucination (Table 3 question).** A "
    "question about the single-attention-head ablation retrieved the *discussion* "
    "prose (section 6.2) instead of Table 3 itself, so `llama3.2:1b` had no real "
    "number to cite and fabricated a plausible-sounding but made-up estimate instead "
    "of saying it didn't know. Mitigation applied: none yet at the model level "
    "(prompt already instructs it to say so if the context lacks the answer, but a "
    "1B model doesn't reliably follow that); the real fix would be retrieval-side -- "
    "e.g. boosting table chunks for questions containing comparison/number language.",
    "2. **Caption vocabulary overlap between similar figures (Figure 4 question).** "
    "A question about anaphora resolution retrieved Figure 3 and Figure 5 instead of "
    "Figure 4, because all three attention-viz captions share heavy vocabulary "
    "(\"attention mechanism\", \"long-distance\", \"relate back\"). The system then "
    "confidently answered with Figure 3's content under a Figure-4-shaped question. "
    "This is the most instructive failure found -- caption-based retrieval alone "
    "isn't enough to disambiguate near-duplicate figures on this paper.",
    "3. **Query phrasing echoing paper boilerplate instead of the technical term "
    "(positional encoding question).** The question paraphrased the paper's own "
    "recurring opening line (\"no recurrence or convolution\"), which pulled dense "
    "similarity toward Background/Introduction/Conclusion (which all open with that "
    "same line) instead of section 3.5, the actual answer. BM25 correctly ranked 3.5 "
    "first, but its poor dense rank dragged it out of the RRF-fused top-4 anyway -- a "
    "concrete example of why hybrid retrieval still isn't foolproof with only k=4.",
    "",
    "General mitigation applied project-wide: hybrid (dense+BM25) retrieval instead "
    "of dense-only, specifically to catch cases like #3; query-time VLM grounding "
    "anchored to the paper's own caption for attention-viz figures, specifically to "
    "reduce hallucination like the source-word mix-up found during figure ingestion "
    "testing (documented separately in CLAUDE.md).",
]

with open("data/processed/evaluation_table.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("Wrote data/processed/evaluation_table.md")
print(f"Hit rate: {len(hits)}/{len(scored)}")
