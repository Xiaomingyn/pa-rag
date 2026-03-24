from fastapi import FastAPI, BackgroundTasks
from .rag import ingest_documents
import os

app = FastAPI()

@app.post("/ingest")
async def ingest(background_tasks: BackgroundTasks):
    data_path = os.getenv("DATA_PATH", "/app/data")
    background_tasks.add_task(ingest_documents, data_path)
    return {"status": "ingestion started", "docs_path": data_path}


# make ingest.py being able to run standalone
if __name__ == "__main__":
    from rag import ingest_documents
    import os
    print(ingest_documents(os.getenv("DATA_PATH")))
