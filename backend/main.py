import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from rag import query_rag
from ingest import app as ingest_app

load_dotenv()

app = FastAPI(title="RAG DeepSeek Backend")
app.mount("/ingest", ingest_app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

@app.post("/query")
def query(body: Query):
    return query_rag(body.question)
