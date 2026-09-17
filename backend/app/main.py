import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes.query import router as query_router
from app.core.config import settings
from app.services.retrieval import RetrievalService
from app.utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading vector store + retrieval index (once, at startup)...")
    app.state.retrieval_service = RetrievalService()
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Attention Is All You Need -- RAG Assistant",
    description="Multimodal RAG backend over the Transformer paper (text, tables, figures).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static/images", StaticFiles(directory=settings.images_dir), name="images")
app.include_router(query_router)
