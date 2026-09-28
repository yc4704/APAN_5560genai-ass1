from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.bigram_model import BigramModel
from app.embedding_model import calculate_embedding


app = FastAPI(
    title="Applied Generative AI API",
    description="A FastAPI service for text generation and word embeddings.",
    version="1.0.0",
)


corpus = [
    (
        "The Count of Monte Cristo is a novel written by "
        "Alexandre Dumas. It tells the story of Edmond Dantes, "
        "who is falsely imprisoned and later seeks revenge."
    ),
    "This is another example sentence.",
    "We are generating text based on bigram probabilities.",
    "Bigram models are simple but effective.",
]


bigram_model = BigramModel(corpus)


class TextGenerationRequest(BaseModel):
    start_word: str = Field(min_length=1)
    length: int = Field(default=20, ge=1, le=100)


class EmbeddingRequest(BaseModel):
    word: str = Field(min_length=1)


@app.get("/")
def read_root():
    return {
        "message": "Applied Generative AI API",
        "documentation": "/docs",
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(
        start_word=request.start_word,
        length=request.length,
    )
    return {"generated_text": generated_text}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    try:
        embedding = calculate_embedding(request.word)
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    return {
        "word": request.word.strip(),
        "dimension": len(embedding),
        "embedding": embedding,
    }