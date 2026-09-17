import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    # `with` triggers the FastAPI lifespan (startup loads the vector store + BM25
    # index once for all tests in this module) -- these are real integration tests
    # that hit the local Ollama daemon, matching how the assignment's own
    # "uvicorn --reload -> test from Swagger UI" verification step works.
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["chunks_loaded"] is True


def test_query_happy_path(client):
    resp = client.post("/query", json={"question": "What is scaled dot-product attention?"})
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["answer"], str) and len(body["answer"]) > 0
    assert isinstance(body["sources"], list) and len(body["sources"]) > 0
    assert isinstance(body["images"], list)


def test_query_invalid_input(client):
    resp = client.post("/query", json={})
    assert resp.status_code == 422


@pytest.mark.parametrize(
    ("greeting", "expected_fragment"),
    [
        ("hi", "Attention Is All You Need"),
        ("hello there", "Attention Is All You Need"),
        ("thanks!", "You're welcome"),
        ("bye", "Goodbye"),
    ],
)
def test_greetings_get_canned_reply_without_retrieval(client, greeting, expected_fragment):
    resp = client.post("/query", json={"question": greeting})
    assert resp.status_code == 200
    body = resp.json()
    assert body["sources"] == []
    assert body["images"] == []
    assert expected_fragment in body["answer"]


def test_identity_question_describes_scope(client):
    resp = client.post("/query", json={"question": "who are you?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["sources"] == []
    assert "Attention Is All You Need" in body["answer"]


@pytest.mark.parametrize(
    "question",
    ["what is the weather in Cairo today?", "tell me a joke", "what is the capital of France?"],
)
def test_off_topic_is_refused_without_citing_sources(client, question):
    resp = client.post("/query", json={"question": question})
    assert resp.status_code == 200
    body = resp.json()
    assert body["sources"] == []
    assert "doesn't appear to be about the paper" in body["answer"]


def test_greeting_prefix_still_answers_the_real_question(client):
    """A greeting in front of a genuine question must not trigger the canned reply."""
    resp = client.post("/query", json={"question": "hi, what is multi-head attention?"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["sources"]) > 0


@pytest.mark.parametrize(
    "greeting",
    ["hiiii", "helloooo", "wasssuppp", "heyyy there", "thx man", "see ya"],
)
def test_misspelled_greeting_variants_handled_by_llm_classifier(client, greeting):
    """Variants the exact-match rules miss must be caught by the LLM classifier
    rather than falling through to an off-topic refusal."""
    resp = client.post("/query", json={"question": greeting})
    assert resp.status_code == 200
    body = resp.json()
    assert body["sources"] == []
    assert "doesn't appear to be about the paper" not in body["answer"]
