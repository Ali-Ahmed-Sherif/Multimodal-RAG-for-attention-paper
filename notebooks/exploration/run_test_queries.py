import json, sys
from rag_pipeline import run_query

TEST_CASES = [
    {"expected_source": "Table 1", "question": (
        "I keep hearing that self-attention is better than RNNs because of path "
        "length. Can you explain what the maximum path length actually is for a "
        "restricted self-attention layer versus a plain recurrent layer, and why "
        "that difference matters?")},
    {"expected_source": "Table 2", "question": (
        "How does the big Transformer model's BLEU score on English-to-French "
        "compare to the ConvS2S Ensemble, and is it actually cheaper to train or "
        "just better?")},
    {"expected_source": "Table 3", "question": (
        "If you cut the model down to just a single attention head instead of the "
        "usual multi-head setup, how much worse does the BLEU score get compared to "
        "the base model?")},
    {"expected_source": "Table 3", "question": (
        "What's the actual difference in perplexity, BLEU, and parameter count "
        "between the base Transformer config and the big one they trained for "
        "longer?")},
    {"expected_source": "Table 4", "question": (
        "For the constituency parsing experiment, how does the semi-supervised "
        "4-layer Transformer's F1 score stack up against the Dyer et al. 2016 "
        "generative model?")},
    {"expected_source": "Figure 1", "question": (
        "I'm looking at the overall architecture diagram and I'm a bit lost on how "
        "the encoder and decoder stacks actually connect to each other. Can you walk "
        "me through the data flow?")},
    {"expected_source": "Figure 2", "question": (
        "Can you explain how Scaled Dot-Product Attention feeds into Multi-Head "
        "Attention based on the diagram? Like, what happens step by step before you "
        "get the final output?")},
    {"expected_source": "Figure 3", "question": (
        "There's a figure showing an attention visualization for a long sentence "
        "about government laws -- what long-distance dependency is it actually "
        "trying to illustrate?")},
    {"expected_source": "Figure 4", "question": (
        "One of the attention figures seems to be about a word referring back to "
        "something earlier in the sentence. What linguistic phenomenon is that, and "
        "which word's attention is being shown?")},
    {"expected_source": "Figure 5", "question": (
        "What's the point of the figure that shows two different attention heads on "
        "the same sentence -- what does it tell us about what the different heads "
        "are doing?")},
    {"expected_source": "Abstract", "question": (
        "According to the abstract, what BLEU score did they get on "
        "English-to-German translation, and roughly how long did it take to train "
        "the model?")},
    {"expected_source": "3.2.1", "question": (
        "Can you explain what scaled dot-product attention actually is, and why "
        "they divide by the square root of d_k instead of just using the raw dot "
        "product?")},
    {"expected_source": "3.2.2", "question": (
        "Why did they bother with multiple attention heads instead of just using "
        "one big attention function? What's the benefit?")},
    {"expected_source": "3.5", "question": (
        "Since the Transformer has no recurrence or convolution at all, how does it "
        "even know the order of the words in a sentence?")},
    {"expected_source": None, "question": (
        "What optimizer did they use to train the model, and did they use a fixed "
        "learning rate or some kind of schedule?")},
    {"expected_source": "7", "question": (  # id for the Conclusion section is "7 Conclusion", not "Conclusion"
        "In the conclusion, what do the authors say are the main advantages of "
        "their approach over the older recurrent and convolutional models?")},
]


def print_report(i, trace):
    print(f"\n{'=' * 90}")
    print(f"[{i}/{len(TEST_CASES)}] Q: {trace['question']}")
    print(f"expected_source={trace['expected_source']}")

    e = trace["embed"]
    print(f"\n-- embed (search_query prefix) -- dims={e['dims']} l2_norm={e['l2_norm']} "
          f"time={e['wall_time_s']}s preview={e['preview']}")

    print("\n-- dense ranking (top 6 of 32, Chroma cosine distance) --")
    for d in trace["dense_full_ranking"][:6]:
        print(f"   {d['id']:>10}  dist={d['distance']}")

    print("-- bm25 ranking (top 6 of 32) --")
    for b in trace["bm25_full_ranking"][:6]:
        print(f"   {b['id']:>10}  score={b['score']}")

    print(f"-- fused top-{len(trace['fused_top_k'])} (RRF) --")
    for f in trace["fused_top_k"]:
        mark = " <-- expected" if f["id"] == trace["expected_source"] else ""
        print(f"   {f['id']:>10}  fused={f['fused_score']}  dense_rank={f['dense_rank']} "
              f"bm25_rank={f['bm25_rank']}  [{f['modality']}] {f['preview']}{mark}")
    print(f"retrieval_hit={trace['retrieval_hit']}")

    if trace["vlm_groundings"]:
        print("\n-- VLM grounding calls (gemma3:4b) --")
        for g in trace["vlm_groundings"]:
            m = g["ollama_metrics"]
            print(f"   [{g['chunk_id']}] image={g['image_path']} time={g['wall_time_s']}s "
                  f"total_duration={m['total_duration_s']}s load_duration={m['load_duration_s']}s "
                  f"prompt_tokens={m['prompt_eval_count']} output_tokens={m['eval_count']}")
            print(f"     response: {g['response'][:300]}{'...' if len(g['response']) > 300 else ''}")

    g = trace["generation"]
    m = g["ollama_metrics"]
    print(f"\n-- generation (llama3.2:1b) -- prompt_chars={trace['final_prompt_chars']} "
          f"time={g['wall_time_s']}s total_duration={m['total_duration_s']}s "
          f"load_duration={m['load_duration_s']}s prompt_tokens={m['prompt_eval_count']} "
          f"output_tokens={m['eval_count']} eval_duration={m['eval_duration_s']}s")
    print(f"ANSWER: {trace['answer']}")

    print(f"\n-- ollama ps before -> after --")
    print(f"   before: {trace['ollama_ps_before']}")
    print(f"   after:  {trace['ollama_ps_after']}")
    print(f"total_wall_time_s={trace['total_wall_time_s']}")


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None  # optional: run a single case index (1-based)
    results = []
    cases = TEST_CASES if not only else [TEST_CASES[int(only) - 1]]
    for i, case in enumerate(cases, start=int(only) if only else 1):
        trace = run_query(case["question"], expected_source=case["expected_source"])
        print_report(i, trace)
        results.append(trace)

    if not only:
        with open("data/processed/eval_trace.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        hits = [r for r in results if r["retrieval_hit"] is True]
        scored = [r for r in results if r["retrieval_hit"] is not None]
        print(f"\n{'=' * 90}")
        print(f"Retrieval hit rate: {len(hits)}/{len(scored)} "
              f"({100 * len(hits) / len(scored):.0f}%)")
        for r in scored:
            if not r["retrieval_hit"]:
                print(f"  MISS: expected {r['expected_source']!r} for: {r['question'][:70]}...")
        print("Wrote data/processed/eval_trace.json")


if __name__ == "__main__":
    main()
