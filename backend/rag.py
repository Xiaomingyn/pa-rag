import os
from typing import List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader  # ✅ Fixed
#from langchain_qdrant import Qdrant                    # ✅ ONLY this Qdrant
from langchain_qdrant import QdrantVectorStore         # ✅ NEW CLASS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from pathlib import Path

load_dotenv()

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))

def get_llm():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("❌ DEEPSEEK_API_KEY missing in .env")
    return ChatOpenAI(
        model="deepseek-chat",  # ✅ More reliable model name
        api_key=api_key,
        base_url="https://api.deepseek.com",
        temperature=0.1
    )

def get_vectorstore():
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    embeddings = get_embeddings()
    
    # Get embedding dimension
    dummy_embedding = embeddings.embed_query("test")
    dim = len(dummy_embedding)
    
    collection_name = os.getenv("INDEX_NAME", "doc_index")
    
    # Create collection if it doesn't exist
    try:
        client.get_collection(collection_name)
        print(f"✅ Collection '{collection_name}' exists")
    except:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )
        print(f"✅ Created collection '{collection_name}' with dim={dim}")
    
    # ✅ FIXED: Use QdrantVectorStore
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings  # Note: embedding (not embeddings)
    )

def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

def ingest_documents(data_path: str = "./data"):
    print(f"📂 Scanning {data_path}...")
    vs = get_vectorstore()
    docs = []
    
    for file_path in Path(data_path).rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in {".pdf", ".docx", ".xlsx", ".txt", ".doc"}:
            print(f"📄 Loading {file_path.name}")
            try:
                loader = UnstructuredLoader(str(file_path), mode="single")
                docs.extend(loader.load())
            except Exception as e:
                print(f"⚠️ Skip {file_path.name}: {e}")
    
    if not docs:
        print("❌ No documents found! Add PDFs/etc to data/")
        return 0
    
    print(f"✂️ Chunking {len(docs)} docs...")
    chunks = chunk_docs(docs)
    print(f"💾 Indexing {len(chunks)} chunks...")
    
    vs.add_documents(chunks)
    print(f"✅ Done! Indexed {len(chunks)} chunks")
    return len(chunks)

def query_rag(question: str, k: int = 5) -> dict:
    try:
        vs = get_vectorstore()
        retriever = vs.as_retriever(search_kwargs={"k": k})
        
        prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant. Answer based ONLY on the context below.

Context: {context}

Question: {question}

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
        sources = [Path(doc.metadata.get("source", "Unknown")).name for doc in docs]
        
        return {"answer": answer, "sources": list(set(sources))}
    except Exception as e:
        print(f"❌ Query error: {e}")
        return {
            "answer": f"Sorry, error occurred: {str(e)}",
            "sources": []
        }
