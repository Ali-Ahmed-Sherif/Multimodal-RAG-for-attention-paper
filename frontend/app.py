import gradio as gr

from api_client import API_BASE_URL, ApiError, ask, check_health, fetch_image

CUSTOM_CSS = """
:root {
    --header-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 45%, #ec4899 100%);
}
.header-banner {
    background: var(--header-gradient);
    border-radius: 18px;
    padding: 28px 32px;
    margin-bottom: 18px;
    box-shadow: 0 10px 30px -12px rgba(99, 102, 241, 0.55);
    animation: fadeSlideIn 0.6s ease-out;
}
.header-banner h1 {
    color: white !important;
    margin: 0 0 6px 0 !important;
    font-size: 1.9rem !important;
}
.header-banner p {
    color: rgba(255,255,255,0.92) !important;
    margin: 0 !important;
}
#chatbot {
    border-radius: 16px !important;
    box-shadow: 0 4px 20px -8px rgba(0,0,0,0.15);
}
#ask_btn {
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
#ask_btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px -4px rgba(99, 102, 241, 0.6);
}
#theme_btn {
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
#theme_btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px -4px rgba(99, 102, 241, 0.45);
}
.example-set button {
    transition: transform 0.12s ease, box-shadow 0.12s ease !important;
}
.example-set button:hover {
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0 4px 12px -4px rgba(99, 102, 241, 0.45);
}
#gallery_panel {
    border-radius: 16px !important;
    animation: fadeSlideIn 0.4s ease-out;
}
#status_badge {
    font-size: 0.85rem;
}
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(-8px); }
    to { opacity: 1; transform: translateY(0); }
}
"""

# Gradio serves light/dark via the ?__theme= query param, so the toggle just reloads
# the page with the opposite value -- no server round-trip needed.
THEME_TOGGLE_JS = """
() => {
    const url = new URL(window.location);
    const current = url.searchParams.get('__theme');
    const isDark = current === 'dark'
        || (current === null && window.matchMedia('(prefers-color-scheme: dark)').matches);
    url.searchParams.set('__theme', isDark ? 'light' : 'dark');
    window.location.href = url.toString();
}
"""

EXAMPLE_QUESTIONS = [
    "How does the big Transformer model's BLEU score on English-to-French compare to the ConvS2S Ensemble?",
    "I'm looking at the overall architecture diagram -- how do the encoder and decoder stacks connect?",
    "One of the attention figures shows a word referring back to something earlier in the sentence. What phenomenon is that?",
    "Why did they use multiple attention heads instead of just one big attention function?",
    "Since the Transformer has no recurrence or convolution, how does it know word order?",
    "What optimizer and learning rate schedule did they use to train the model?",
]

THEME = gr.themes.Soft(primary_hue="indigo", secondary_hue="pink", neutral_hue="slate")


def format_sources(sources: list[str]) -> str:
    return "  ".join(f"`[{s}]`" for s in sources)


def respond(question, history):
    history = history or []
    question = (question or "").strip()
    if not question:
        yield history, gr.update(visible=False), ""
        return

    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": "_Searching the paper and generating an answer..._"},
    ]
    yield history, gr.update(visible=False), ""

    try:
        result = ask(question)
    except ApiError as e:
        history[-1] = {"role": "assistant", "content": f"**Error:** {e}"}
        yield history, gr.update(visible=False), ""
        return
    except Exception as e:
        history[-1] = {"role": "assistant", "content": f"**Unexpected error:** {e}"}
        yield history, gr.update(visible=False), ""
        return

    answer = result.get("answer", "")
    sources = result.get("sources", [])
    images = result.get("images", [])

    text = answer
    if sources:
        text += f"\n\n---\n**Sources:** {format_sources(sources)}"
    history[-1] = {"role": "assistant", "content": text}

    gallery_items = []
    for img in images:
        if not img.get("url"):
            continue
        pil_image = fetch_image(img["url"])
        if pil_image is not None:
            gallery_items.append((pil_image, f"{img['chunk_id']} - {img['caption'][:90]}"))
    gallery_update = gr.update(value=gallery_items, visible=bool(gallery_items))
    yield history, gallery_update, ""


def refresh_status():
    if check_health():
        return "**Status:** connected to backend"
    return f"**Status:** backend unreachable at {API_BASE_URL} - start it with `uvicorn app.main:app`"


with gr.Blocks(title="Attention RAG Assistant") as demo:
    gr.HTML(
        """
        <div class="header-banner">
            <h1>Attention Is All You Need - RAG Assistant</h1>
            <p>Ask a question about the Transformer paper. Answers are grounded in the
            actual text, tables, and figures of the paper, including vision-model
            grounding on the architecture diagrams and attention visualizations.</p>
        </div>
        """
    )

    with gr.Row():
        status_badge = gr.Markdown(elem_id="status_badge")
        theme_btn = gr.Button("Switch light / dark", size="sm", scale=0, elem_id="theme_btn")
    theme_btn.click(fn=None, inputs=None, outputs=None, js=THEME_TOGGLE_JS)
    demo.load(fn=refresh_status, outputs=status_badge)

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                height=480,
                elem_id="chatbot",
                label="Conversation",
            )
            with gr.Row():
                question_box = gr.Textbox(
                    placeholder="e.g. What is scaled dot-product attention?",
                    show_label=False,
                    scale=5,
                )
                ask_btn = gr.Button("Ask", variant="primary", scale=1, elem_id="ask_btn")

            gr.Examples(
                examples=EXAMPLE_QUESTIONS,
                inputs=question_box,
                label="Try one of these",
                elem_id="example_set",
            )

        with gr.Column(scale=1):
            gr.Markdown("### Cited figures")
            gallery = gr.Gallery(
                visible=False,
                elem_id="gallery_panel",
                label=None,
                show_label=False,
                columns=1,
                height=520,
                object_fit="contain",
            )

    ask_btn.click(respond, [question_box, chatbot], [chatbot, gallery, question_box])
    question_box.submit(respond, [question_box, chatbot], [chatbot, gallery, question_box])

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CUSTOM_CSS)
