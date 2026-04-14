import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import pipeline
from schemas import CampaignBrief, GenerateResponse

load_dotenv()

app = FastAPI(title="Campaign Strategy Backend")

env_origins = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "").split(",") if origin.strip()]
allowed_origins = ["http://localhost:3000", *env_origins]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"error": "Pipeline failed", "detail": str(exc)},
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(brief: CampaignBrief) -> GenerateResponse:
    result = pipeline.run(brief)
    return GenerateResponse(**result)
