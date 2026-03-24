import os
from typing import List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_unstructured import UnstructuredLoader
from pathlib import Path

load_dotenv()

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL"))

def get_llm():
    return ChatOpenAI(
        model="deepseek-reasoner",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=0.1
    )

def get_vectorstore():
    client = QdrantClient(url=os.getenv("QDRANT_URL"))
    embeddings = get_embeddings()
    
    # Get embedding dimension
    dummy_embedding = embeddings.embed_query("test")
    dim = len(dummy_embedding)
    
    # Create collection if it doesn't exist
    try:
        client.get_collection(os.getenv("INDEX_NAME"))
        print(f"✅ Collection '{os.getenv('INDEX_NAME')}' exists")
    except:
        client.create_collection(
            collection_name=os.getenv("INDEX_NAME"),
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )
        print(f"✅ Created collection '{os.getenv('INDEX_NAME')}' with dim={dim}")
    
    return Qdrant(
        client=client,
        collection_name=os.getenv("INDEX_NAME"),
        embeddings=embeddings
    )
    
def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

def ingest_documents(data_path: str):
    print(f"📂 Scanning {data_path}...")
    vs = get_vectorstore()
    docs = []
    
    for file_path in Path(data_path).rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in {".pdf", ".docx", ".xlsx", ".txt", ".doc"}:
            print(f"📄 Loading {file_path.name}")
            try:
                loader = UnstructuredLoader(str(file_path))
                docs.extend(loader.load())
            except Exception as e:
                print(f"⚠️  Skip {file_path.name}: {e}")
    
    if not docs:
        print("❌ No documents found! Check your data/ folder.")
        return 0
    
    print(f"✂️  Chunking {len(docs)} docs...")
    chunks = chunk_docs(docs)
    print(f"💾 Indexing {len(chunks)} chunks...")
    
    vs.add_documents(chunks)
    print(f"✅ Done! Indexed {len(chunks)} chunks")
    return len(chunks)

def query_rag(question: str, k: int = 5) -> dict:
    vs = get_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": k})
    
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on context:
        Context: {context}
        Question: {question}
        Sources: {sources}
        
        Answer:"""
    )
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | get_llm()
        | StrOutputParser()
    )
    
    answer = chain.invoke(question)
    
    # Get sources
    docs = retriever.invoke(question)
    sources = [doc.metadata.get("source", "Unknown") for doc in docs]
    
    return {"answer": answer, "sources": list(set(sources))}