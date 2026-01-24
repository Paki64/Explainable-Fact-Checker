import yaml
from langchain_community.document_loaders import CSVLoader, DirectoryLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from pymongo import MongoClient
from tqdm import tqdm
import json
from urllib.parse import quote_plus


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


def create_vector(chunks):

    print("Creating vector store...")
    embeddings = OllamaEmbeddings(model=config["embeddings_model"])
    collection = db_connect()
    
    embedded_chunks = []
    with tqdm(total=len(chunks), desc="Embedding chunks", unit="chunk") as pbar:
        for chunk in chunks:
            embedding = embeddings.embed_query(chunk.page_content)
            
            doc = {
                "content": chunk.page_content,
                "embedding": embedding,
                "metadata": chunk.metadata if hasattr(chunk, 'metadata') else {}
            }
            
            collection.insert_one(doc)
            embedded_chunks.append(chunk)
            pbar.update(1)
    
    print(f"Vector DB created with {len(embedded_chunks)} chunks in MongoDB")
    return collection


if __name__ == "__main__":
    documents = load_documents(config["file_path"])
    chunks = split_documents(documents)
    vector_store = create_vector(chunks)