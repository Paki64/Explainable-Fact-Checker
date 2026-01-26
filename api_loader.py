import yaml
from langchain_community.document_loaders import CSVLoader, DirectoryLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pymongo import MongoClient
from tqdm import tqdm
from urllib.parse import quote_plus
import os
from huggingface_hub import InferenceClient
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


def load_documents(data_path: str):

    documents = []
    
    # CSV files
    print("Loading CSV documents...")
    loader = DirectoryLoader(
        path=data_path,
        glob="**/*.csv",
        loader_cls=CSVLoader,
        show_progress=True,
    )
    csv_documents = loader.load()
    documents.extend(csv_documents)
    print(f"Loaded {len(csv_documents)} CSV documents.")

    '''
    # JSONL files
    print("Loading JSONL documents...")
    loader_kwargs = {
        "jq_schema": ".",          
        "content_key": "text",     
        "json_lines": True,        
        "text_content": True       
    }
    loader = DirectoryLoader(
        path=data_path,
        glob="**/*.jsonl",         
        loader_cls=JSONLoader,
        loader_kwargs=loader_kwargs,
        show_progress=True        
    )
    jsonl_documents = loader.load()
    documents.extend(jsonl_documents)
    print(f"Loaded {len(jsonl_documents)} JSONL documents.")
    '''
    
    if len(documents) == 0:
        raise FileNotFoundError("No documents found in the specified directory.")
    print(f"Total loaded: {len(documents)} documents.")
    return documents


def split_documents(documents):

    batch_size = config["batch_size"]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        separators=["\n\n---\n\n", "\n\n", "\n", " ", ""],
        length_function=len,
    )
    all_chunks = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        combined_text = "\n\n---\n\n".join([doc.page_content for doc in batch])
        chunk_texts = text_splitter.split_text(combined_text)
        batch_chunks = [Document(page_content=chunk) for chunk in chunk_texts]
        all_chunks.extend(batch_chunks)
        del combined_text, chunk_texts, batch_chunks

    chunks = all_chunks
    print(f"Document split into {len(chunks)} chunks.")
    return chunks


def db_connect():
    print("Connecting to MongoDB...")

    uri = config["mongodb_uri"]
    username = config.get("mongodb_username", "")
    password = config.get("mongodb_password", "")
    auth_db = config.get("mongodb_auth_db", "admin")
    
    if username and password:
        escaped_username = quote_plus(username)
        escaped_password = quote_plus(password)
        uri = uri.replace("mongodb://", f"mongodb://{escaped_username}:{escaped_password}@")
        client = MongoClient(uri, authSource=auth_db, serverSelectionTimeoutMS=5000)
    else:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    
    try:
        client.admin.command('ping')
        print("Connected to MongoDB.")
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        print("Check your credentials and MongoDB is running")
        raise
    
    db = client[config["mongodb_db"]]
    collection = db[config["mongodb_collection"]]
    return collection


# Thread-safe client cache
_client_cache = threading.local()

def get_hf_client():
    """Thread-safe HF client"""
    if not hasattr(_client_cache, 'client'):
        _client_cache.client = InferenceClient(
            provider="hf-inference",
            api_key=os.getenv("HF_TOKEN"),
        )
    return _client_cache.client

def hf_batch(texts, model_name):
    """Single batch embedding - ottimizzata"""
    client = get_hf_client()
    try:
        return client.feature_extraction(texts, model=model_name)
    except Exception as e:
        if "504" in str(e):
            time.sleep(1)  # Breve pausa
            return client.feature_extraction(texts, model=model_name)
        raise e

def create_vector(chunks):
    print("🚀 Creating vector store - OPTIMIZED HF API")
    
    model_name =  "sentence-transformers/all-MiniLM-L6-v2"
    collection = db_connect()
    
    BATCH_SIZE = 64  # Ottimale per MiniLM
    buffer_size = config.get("buffer_chunks", 1000)
    
    print(f"Model: {model_name} | Chunks: {len(chunks)} | Batch: {BATCH_SIZE}")
    start_time = time.time()
    
    # 1. Prepara batches
    batches = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_texts = [chunk.page_content for chunk in chunks[i:i+BATCH_SIZE]]
        batches.append((batch_texts, i))
    
    # 2. Parallelismo: 8 workers max
    all_embeddings = [None] * len(chunks)
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(hf_batch, texts, model_name): (idx, start, end)
            for idx, (texts, start) in enumerate(batches)
            for end in [start + len(texts)]
        }
        
        for future in tqdm(as_completed(futures), total=len(batches), desc="⚡ Parallel API"):
            try:
                batch_embs = future.result(timeout=120)
                batch_idx, start, end = futures[future]
                all_embeddings[start:end] = batch_embs
            except Exception as e:
                print(f"⚠️ Batch failed: {e}")
    
    elapsed = time.time() - start_time
    print(f"⏱️  Embeddings: {elapsed:.1f}s ({len(chunks)/elapsed:.0f} chunks/s)")
    
    # 3. MongoDB bulk insert superveloce
    buffered_docs = []
    for chunk, emb in zip(chunks, all_embeddings):
        doc = {
            "content": chunk.page_content,
            "embedding": emb.tolist() if hasattr(emb, "tolist") else list(emb),
            "metadata": getattr(chunk, "metadata", {}),
        }
        buffered_docs.append(doc)
        
        if len(buffered_docs) >= buffer_size:
            collection.insert_many(buffered_docs, ordered=False)
            buffered_docs = []
    
    if buffered_docs:
        collection.insert_many(buffered_docs, ordered=False)
    
    total_time = time.time() - start_time
    print(f"✅ FINISHED: {len(chunks)} chunks in {total_time:.1f}s | {len(chunks)/total_time:.0f} chunks/s")
    return collection

if __name__ == "__main__":
    documents = load_documents(config["file_path"])
    chunks = split_documents(documents)
    vector_store = create_vector(chunks)
