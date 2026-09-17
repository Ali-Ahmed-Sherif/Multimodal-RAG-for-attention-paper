from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # absolute so backend/.env is found no matter which directory the server is
    # started from (e.g. `uvicorn --app-dir backend` run from the repo root)
    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    ollama_host: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    vlm_model: str = "gemma3:4b"
    llm_model: str = "llama3.2:1b"

    vector_store_path: str = str(BACKEND_ROOT / "data" / "vector_store")
    chunks_path: str = str(BACKEND_ROOT / "data" / "all_chunks_embedded.json")
    images_dir: str = str(BACKEND_ROOT / "data" / "images")

    top_k: int = 4
    rrf_k: int = 60
    # best cosine distance above which a question is treated as off-topic -- see the
    # measured on-topic vs off-topic distributions in app/services/intent.py
    scope_distance_threshold: float = 0.40

    cors_origins: list[str] = ["http://localhost:7860", "http://127.0.0.1:7860"]

    log_level: str = "INFO"


settings = Settings()
