import logging

from fastapi import APIRouter, HTTPException, Request

from app.schemas.query import QueryRequest, QueryResponse
from app.services.generation import generate_answer
from app.services.intent import (
    OUT_OF_SCOPE_REPLY,
    REPLY_FOR_INTENT,
    classify_intent,
    is_out_of_scope,
    match_canned,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
def health(request: Request):
    ready = getattr(request.app.state, "retrieval_service", None) is not None
    return {"status": "ok" if ready else "starting", "chunks_loaded": ready}


@router.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, request: Request):
    # guard rail 1 (~0ms): common spellings of greetings/identity/thanks, no model call
    canned = match_canned(body.question)
    if canned is not None:
        logger.info("canned reply for %r", body.question[:60])
        return QueryResponse(answer=canned, sources=[], images=[])

    retrieval_service = request.app.state.retrieval_service
    try:
        # guard rail 2 (~100ms): LLM catches variants the rules miss ("wasssuppp")
        intent = classify_intent(body.question)
        if intent in REPLY_FOR_INTENT:
            return QueryResponse(answer=REPLY_FOR_INTENT[intent], sources=[], images=[])

        chunks, best_distance = retrieval_service.hybrid_retrieve(body.question)

        # guard rail 3: nothing in the paper is close enough -- answer without
        # burning a VLM/LLM call on a question the corpus can't support
        if is_out_of_scope(best_distance):
            logger.info("out of scope (distance=%.4f) for %r", best_distance, body.question[:60])
            return QueryResponse(answer=OUT_OF_SCOPE_REPLY, sources=[], images=[])

        result = generate_answer(chunks, body.question)
    except Exception:
        logger.exception("query failed for question=%r", body.question)
        raise HTTPException(status_code=502, detail="Failed to generate an answer. Is Ollama running?")
    return QueryResponse(**result)
