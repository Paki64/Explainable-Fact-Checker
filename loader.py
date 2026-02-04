import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm
import json
from pathlib import Path
from db_connect import db_connect


with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


# Normalizzazione testo
def _normalize_text(value):

    # Rimuovi spazi bianchi nelle stringhe
    if isinstance(value, str):
        return value.strip() 
    
    # Unisci elementi di lista in una singola stringa
    if isinstance(value, list):
        parts = [v for v in value if isinstance(v, str) and v.strip()]
        return "\n".join(parts).strip() if parts else ""
    
    return ""


# Estrai testo in base a chiavi comuni
def _extract_text(row: dict) -> str:
    
    for key in ("text", "content", "body", "article", "claim", "question", "url2text", "lines"):
        text = _normalize_text(row.get(key))
        if text:
            return text
        
    return ""


# Carica documenti da file JSON
def load_documents(data_path: str):
    documents = []
    json_files = sorted(Path(data_path).rglob("*.json"))
    
    for json_file in tqdm(json_files, desc="Loading JSON files"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                content = f.read()
                # Prova a caricare come JSON completo o come linee JSON in base al formato del file
                try:
                    data = json.loads(content)
                    rows = data if isinstance(data, list) else [data]
                except json.JSONDecodeError:
                    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
                # Estrai testo da ogni riga
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    text = _extract_text(row)
                    if not text:
                        continue
                    # Aggiunta dei metadati
                    metadata = {"url": row.get("url", "")}
                    documents.append(Document(page_content=text, metadata=metadata))
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    # Eccezione se nessun documento trovato
    if not documents:
        raise FileNotFoundError("No documents found.")
    
    print(f"Loaded {len(documents)} documents.")
    return documents


# Suddivisione dei documenti in chunk
def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        separators=["\n\n---\n\n", "\n\n", "\n", " ", ""],
        length_function=len,
    )

    all_chunks = []
    for doc in tqdm(documents, desc="Splitting documents"):
        chunk_texts = text_splitter.split_text(doc.page_content)
        for chunk_text in chunk_texts:
            all_chunks.append(Document(page_content=chunk_text, metadata=doc.metadata.copy()))

    print(f"Split into {len(all_chunks)} chunks.")
    return all_chunks


# Creazione del database vettoriale e caricamento degli embeddings su DB
def create_vector(chunks):

    embeddings = OllamaEmbeddings(model=config["embeddings_model"])
    collection = db_connect()
    batch_size = config.get("batch_size", 32)

    with tqdm(total=len(chunks), desc="Embedding chunks") as pbar:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [c.page_content for c in batch]
            embeddings_batch = embeddings.embed_documents(texts)

            docs = [
                {
                    "content": chunk.page_content,
                    "embedding": embedding,
                    "metadata": chunk.metadata,
                }
                for chunk, embedding in zip(batch, embeddings_batch)
            ]
            collection.insert_many(docs, ordered=False)
            pbar.update(len(batch))

    print(f"Created vector DB with {len(chunks)} chunks")
    return collection



if __name__ == "__main__":
    documents = load_documents(config["file_path"])
    chunks = split_documents(documents)
    vector_store = create_vector(chunks)