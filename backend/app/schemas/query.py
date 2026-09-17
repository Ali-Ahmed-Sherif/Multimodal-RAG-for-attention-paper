from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question about the paper")


class ImageSource(BaseModel):
    chunk_id: str
    caption: str
    url: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    images: list[ImageSource] = []
