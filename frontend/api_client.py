import io
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from PIL import Image

# absolute so frontend/.env is found even when launched from the repo root
# (e.g. `python frontend/app.py`)
load_dotenv(Path(__file__).resolve().parent / ".env")

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
TIMEOUT = float(os.environ.get("API_TIMEOUT_SECONDS", "120"))


class ApiError(Exception):
    """Raised for any backend failure the UI should show as a friendly message."""


def check_health() -> bool:
    try:
        resp = httpx.get(f"{API_BASE_URL}/health", timeout=5)
        return resp.status_code == 200 and resp.json().get("status") == "ok"
    except httpx.HTTPError:
        return False


def ask(question: str) -> dict:
    """Calls POST /query. Returns {'answer', 'sources', 'images'}. Raises ApiError
    with a user-friendly message on any failure."""
    try:
        resp = httpx.post(f"{API_BASE_URL}/query", json={"question": question}, timeout=TIMEOUT)
    except httpx.ConnectError as e:
        raise ApiError(f"Can't reach the backend at {API_BASE_URL}. Is it running?") from e
    except httpx.TimeoutException as e:
        raise ApiError("The backend took too long to respond (model may still be loading). Try again.") from e

    if resp.status_code == 422:
        raise ApiError("That question wasn't accepted by the backend. Try rephrasing it.")
    if resp.status_code >= 500:
        raise ApiError("The backend hit an error generating the answer. Check that Ollama is running.")
    resp.raise_for_status()
    return resp.json()


def fetch_image(relative_url: str) -> Image.Image | None:
    """Downloads an image the backend referenced (e.g. '/static/images/figure1.png')
    and returns it as a PIL Image. Gradio's Gallery proxies/validates URL sources
    server-side and blocks cross-origin fetches, so we fetch the bytes ourselves and
    hand Gradio real image data instead of a URL."""
    try:
        resp = httpx.get(f"{API_BASE_URL}{relative_url}", timeout=10)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content))
    except (httpx.HTTPError, OSError):
        return None
