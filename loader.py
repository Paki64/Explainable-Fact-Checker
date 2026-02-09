import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm
import json
from pathlib import Path
from db_connect import db_connect


with open("config/config.yaml", "r") as file:
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


# Estrazione del testo 
def _extract_text(row: dict) -> str:
    
    # Da applicare con chiavi comuni
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

                # Caricamento e decodifica del JSON
                try:
                    data = json.loads(content)                         
                    rows = data if isinstance(data, list) else [data]  # JSON completo o riga per riga
                except json.JSONDecodeError:
                    rows = [json.loads(line) for line in content.splitlines() if line.strip()] # Fallback: decodifica riga per riga
                
                # Estrazione del testo
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

    # [ECCEZIONE] Documento non trovato
    if not documents:
        raise FileNotFoundError("No documents found.")
    
    print(f"Loaded {len(documents)} documents.")
    return documents


# Suddivisione dei documenti in chunk
def split_documents(documents):
    chunk_size = config["chunk_size"]
    chunk_overlap = config["chunk_overlap"]
    
    # Suddivisione del testo specifico   
    def _split_text_strict(text: str) -> list[str]:
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        step = max(1, chunk_size - chunk_overlap)
        
        for i in range(0, len(text), step):
            chunk = text[i:i + chunk_size]
            chunks.append(chunk)
            if i + chunk_size >= len(text):
                break
        
        return chunks

    all_chunks = []
    max_chunk_size = 0
    total_chars = 0
    
    for doc in tqdm(documents, desc="Splitting documents"):
        text = doc.page_content
        total_chars += len(text)
        chunk_texts = _split_text_strict(text)
        
        for chunk_text in chunk_texts:
            all_chunks.append(Document(page_content=chunk_text, metadata=doc.metadata.copy()))
            chunk_len = len(chunk_text)
            
            # Gestione del chunk più grande
            max_chunk_size = max(max_chunk_size, chunk_len)
            if chunk_len > chunk_size:
                raise ValueError(f"Chunk size {chunk_len} exceeds limit {chunk_size}")

    avg_chunk_size = total_chars / len(all_chunks) if all_chunks else 0
    print(f"\nTotal text: {total_chars:,} chars")
    print(f"Split into {len(all_chunks)} chunks.")
    print(f"Average chunk size: {avg_chunk_size:.0f} chars")

    return all_chunks


# Quantizzazione degli embeddings da Float32 a INT8
def quantize_embedding(embedding: list) -> list:
    import numpy as np
    emb_array = np.array(embedding, dtype=np.float32)
    emb_normalized = emb_array / (np.linalg.norm(emb_array) + 1e-8)
    return (emb_normalized * 127).astype(np.int8).tolist()


# Creazione del database vettoriale e caricamento degli embeddings su DB
def create_vector(chunks):
    import time # [DEBUG] Misurazione del tempo di esecuzione
    import numpy as np

    # Configurazione del modello di embedding
    embeddings = OllamaEmbeddings(model=config["embeddings_model"])
    collection = db_connect()
    
    # Gestione dei chunks troppo grandi
    def _split_oversized_chunk(text: str, metadata: dict) -> list[Document]:
        max_size = config["max_size"] 
        if len(text) <= max_size:
            return [Document(page_content=text, metadata=metadata)]
        
        pieces = []
        for i in range(0, len(text), max_size):
            piece = text[i:i + max_size]
            pieces.append(Document(page_content=piece, metadata=metadata.copy()))
        return pieces
    
    total_inserted = 0
    total_split = 0
    errors = 0
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"Starting embedding process for {len(chunks)} chunks")
    print(f"{'='*60}\n")
    
    for idx, chunk in enumerate(tqdm(chunks, desc="Embedding chunks"), start=1):
        chunk_size = len(chunk.page_content)
        
        try:
            embedding = embeddings.embed_documents([chunk.page_content])
            embedding_quantized = quantize_embedding(embedding[0])  # [HACK] Quantizzazione a INT8 per risparmiare spazio su DB
            
            doc = {
                "content": chunk.page_content,
                "embedding": embedding_quantized,
                "metadata": chunk.metadata,
            }
            collection.insert_one(doc)
            total_inserted += 1
            
            # [DEBUG] Report di progresso
            if idx % 100 == 0:
                elapsed = time.time() - start_time
                rate = idx / elapsed
                eta = (len(chunks) - idx) / rate if rate > 0 else 0
                avg_size = sum(len(c.page_content) for c in chunks[:idx]) / idx
                print(f"\n[{idx}/{len(chunks)}] Rate: {rate:.2f} chunks/s | "
                      f"ETA: {eta/60:.1f}m | Avg size: {avg_size:.0f} chars | "
                      f"Inserted: {total_inserted} | Split: {total_split}")
                
        # [ECCEZIONE] Gestione dei chunk troppo grandi
        except Exception as e:
            if "exceeds the context length" in str(e):
                total_split += 1
                print(f"\n[WARNING] Chunk {idx} too large ({chunk_size} chars), splitting...")
                
                # I chunk troppo grandi vengono risuddivisiO
                sub_chunks = _split_oversized_chunk(chunk.page_content, chunk.metadata)
                print(f"  → Split into {len(sub_chunks)} sub-chunks")
                
                for sub_idx, sub_chunk in enumerate(sub_chunks, start=1):
                    try:
                        sub_embedding = embeddings.embed_documents([sub_chunk.page_content])
                        sub_embedding_quantized = quantize_embedding(sub_embedding[0])
                        doc = {
                            "content": sub_chunk.page_content,
                            "embedding": sub_embedding_quantized,
                            "metadata": sub_chunk.metadata,
                        }
                        collection.insert_one(doc)
                        total_inserted += 1
                    except Exception as sub_e:
                        errors += 1
                        sub_size = len(sub_chunk.page_content)
                        print(f"  [ERROR] Sub-chunk {sub_idx} ({sub_size} chars) failed: {sub_e}")
            else:
                errors += 1
                print(f"\n[ERROR] Chunk {idx} failed with unexpected error: {e}")
                print(f"  Chunk size: {chunk_size} chars")
                print(f"  URL: {chunk.metadata.get('url', 'N/A')}")

    # [DEBUG] Report finale
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Embedding completed!")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Average rate: {len(chunks)/total_time:.2f} chunks/s")
    print(f"Successfully inserted: {total_inserted} documents")
    print(f"Chunks split: {total_split}")
    print(f"Errors: {errors}")
    print(f"{'='*60}\n")
    
    return collection



if __name__ == "__main__":
    documents = load_documents(config["file_path"])
    chunks = split_documents(documents)
    vector_store = create_vector(chunks)