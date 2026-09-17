"""Guard rails in front of the RAG pipeline, so greetings and off-topic input don't
run the full embed -> retrieve -> VLM -> LLM workflow.

Three layers, each doing the job it was measured to be good at:

  1. `match_canned()`   -- exact/regex rules. ~0ms, zero model calls. Covers the common
                           spellings ("hi", "thanks", "who are you?").
  2. `classify_intent()`-- llama3.2:1b with a constrained JSON schema, ~100ms. Catches
                           arbitrary variants the rules miss: "hiiii", "helloooo",
                           "wassup", "wasssuppp", "heyyy there", "ok cool".
  3. `is_out_of_scope()`-- embedding distance vs. a threshold, decided AFTER retrieval.

Why the split (all measured -- see CLAUDE.md for the numbers): a 3-label prompt
(GREETING/IDENTITY/QUESTION) scored 89% with 12/12 greetings and 18/18 questions right.
An earlier 4-label version that also asked the model whether a question was *about the
paper* scored 48-58% and mislabelled 9 genuine paper questions as off-topic -- refusing
to answer real questions, the worst possible failure. Judging topicality needs knowledge
of the corpus, so that decision belongs to the embedding distance, not the 1B model.
"""

import json
import logging
import re

import ollama

from app.core.config import settings

logger = logging.getLogger(__name__)

GREETING = "GREETING"
IDENTITY = "IDENTITY"
QUESTION = "QUESTION"

SCOPE_BLURB = (
    "I answer questions about the paper \"Attention Is All You Need\" (Vaswani et al., "
    "2017). You can ask about the Transformer architecture, the attention mechanisms, "
    "positional encoding, the training setup, the results tables, or any of the figures."
)

GREETING_REPLY = f"Hello. {SCOPE_BLURB}"
IDENTITY_REPLY = (
    f"I'm a retrieval-augmented assistant for a single document. {SCOPE_BLURB} "
    "Answers are grounded in the paper's actual text, tables and figures, and cite the "
    "section or table they came from rather than relying on the model's own knowledge."
)
THANKS_REPLY = "You're welcome. Ask me anything else about the paper."
FAREWELL_REPLY = "Goodbye."
OUT_OF_SCOPE_REPLY = (
    f"That question doesn't appear to be about the paper, so I can't answer it from my "
    f"sources. {SCOPE_BLURB}"
)

REPLY_FOR_INTENT = {GREETING: GREETING_REPLY, IDENTITY: IDENTITY_REPLY}

GREETINGS = {
    "hi", "hii", "hiya", "hello", "helo", "hey", "yo", "greetings", "salam", "salamu alaikum",
    "hi there", "hello there", "hey there", "good morning", "good afternoon", "good evening",
    "howdy", "sup", "wassup", "whats up", "how are you", "how are you doing",
}
THANKS = {
    "thanks", "thank you", "thanks a lot", "thanks so much", "thank you very much",
    "ty", "thx", "appreciate it", "appreciated", "nice", "cool", "great", "awesome",
    "perfect", "ok", "okay", "got it", "ok cool",
}
FAREWELLS = {"bye", "goodbye", "good bye", "see you", "see ya", "cya", "later", "good night"}

IDENTITY_RE = re.compile(
    r"^(who|what)\s+(are|r)\s+(you|u)\b"
    r"|^what\s+(can|do)\s+(you|u)\s+do\b"
    r"|^what\s+(is|are)\s+(this|you)\s+for\b"
    r"|^what\s+is\s+this\b"
    r"|^how\s+do\s+(you|u)\s+work\b"
    r"|^what\s+can\s+i\s+ask\b"
    r"|^introduce\s+yourself\b"
    r"|^help$"
)

CLASSIFIER_SCHEMA = {
    "type": "object",
    "properties": {"intent": {"type": "string", "enum": [GREETING, IDENTITY, QUESTION]}},
    "required": ["intent"],
}

# Deliberately free of paper vocabulary: an earlier prompt stuffed with Transformer
# terms biased the 1B model into labelling everything as a paper question.
CLASSIFIER_SYSTEM = """You label a single user message with one of three labels.

GREETING = a greeting, small talk, thanks, or goodbye. No real question is asked.
IDENTITY = the user asks what you are, who you are, or what you can do.
QUESTION = the user asks a real question about any subject.

Label only the message given. Do not answer it. Reply with JSON."""

CLASSIFIER_FEWSHOT = [
    ("hello there", GREETING),
    ("thx man", GREETING),
    ("bye", GREETING),
    ("yo!", GREETING),
    ("what can you do?", IDENTITY),
    ("how do you work?", IDENTITY),
    ("What is multi-head attention?", QUESTION),
    ("what is the capital of France?", QUESTION),
    ("how many encoder layers are there?", QUESTION),
]


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()


def match_canned(question: str) -> str | None:
    """Fast path: a fixed reply for the common spellings, or None to keep going.

    Only fires on exact matches or short inputs, so a real question that merely opens
    with a greeting ("hi, why do they scale by sqrt(d_k)?") still reaches retrieval.
    """
    normalized = _normalize(question)
    if not normalized:
        return GREETING_REPLY
    if normalized in GREETINGS:
        return GREETING_REPLY
    if normalized in THANKS:
        return THANKS_REPLY
    if normalized in FAREWELLS:
        return FAREWELL_REPLY
    if len(normalized.split()) <= 8 and IDENTITY_RE.search(normalized):
        return IDENTITY_REPLY
    return None


def classify_intent(question: str) -> str:
    """Asks the LLM for GREETING / IDENTITY / QUESTION.

    Fails safe: any Ollama or parsing error returns QUESTION so the normal RAG path
    still runs -- a classifier outage must not stop real questions being answered.
    """
    messages = [{"role": "system", "content": CLASSIFIER_SYSTEM}]
    for text, label in CLASSIFIER_FEWSHOT:
        messages.append({"role": "user", "content": text})
        messages.append({"role": "assistant", "content": json.dumps({"intent": label})})
    messages.append({"role": "user", "content": question})

    try:
        resp = ollama.chat(
            model=settings.llm_model,
            messages=messages,
            format=CLASSIFIER_SCHEMA,
            options={"temperature": 0},
        )
        intent = json.loads(resp.message.content)["intent"]
    except Exception:
        logger.exception("intent classification failed, defaulting to QUESTION")
        return QUESTION

    if intent not in (GREETING, IDENTITY, QUESTION):
        logger.warning("classifier returned unexpected intent %r, defaulting to QUESTION", intent)
        return QUESTION
    logger.info("classified %r as %s", question[:60], intent)
    return intent


def is_out_of_scope(best_distance: float) -> bool:
    return best_distance > settings.scope_distance_threshold
