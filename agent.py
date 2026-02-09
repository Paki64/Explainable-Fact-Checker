import json
import re
import time
import yaml
import numpy as np
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from db_connect import db_connect

with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

with open("config/prompt.yaml", "r") as file:
    prompt = yaml.safe_load(file)


_EMBEDDINGS_CACHE = None

# Carica la cache degli embeddings in memoria
def load_embeddings_cache():
    global _EMBEDDINGS_CACHE
    
    if _EMBEDDINGS_CACHE is not None:
        return _EMBEDDINGS_CACHE
    
    t0 = time.time()
    
    # Connessione al database e caricamento degli embeddings
    print("[INIT] Connessione al DB in corso...")
    collection = db_connect()
    print("[INIT] Caricamento embeddings in memoria...")
    cursor = collection.find({}, {"embedding": 1, "_id": 1})
    
    doc_ids = []
    embeddings_list = []
    
    # Iterazione sui documenti per estrarre gli embeddings e gli ID
    for doc in cursor:
        if "embedding" in doc:
            doc_ids.append(doc["_id"])
            # [HACK] Conversione degli embeddings da int8 a float32
            emb_int8 = np.array(doc["embedding"], dtype=np.int8)
            emb_float = emb_int8.astype(np.float32) / 127.0
            embeddings_list.append(emb_float.tolist())
    
    # Conversione in matrice NumPy
    embeddings_matrix = np.array(embeddings_list, dtype=np.float32)
    
    # Libera memoria 
    del embeddings_list # Lista originale non più necessaria
    import gc           # Garbage collector
    gc.collect()
    
    # Debug: controlla dimensioni
    num_docs = len(doc_ids)
    memory_mb = embeddings_matrix.nbytes / (1024 * 1024)
    
    # Salva nella cache globale
    _EMBEDDINGS_CACHE = {
        "doc_ids": doc_ids,
        "embeddings_matrix": embeddings_matrix,
        "collection": collection
    }
    
    t1 = time.time()
    print(f"[INIT] Cache pronta: {num_docs} embeddings in {t1-t0:.2f}s (~{memory_mb:.0f} MB RAM)")
    
    return _EMBEDDINGS_CACHE


# Calcolo della similarità coseno tra due vettori
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# Retrieving dei documenti più rilevanti in base alla similarità coseno
def retrieve_relevant_info(question: str, min_score: float = 0.35, top_k: int = 5):
    
    # Usa la cache invece di ricaricare ogni volta
    cache = load_embeddings_cache()
    embeddings_matrix = cache["embeddings_matrix"]
    doc_ids = cache["doc_ids"]
    collection = cache["collection"]
    
    # Connessione agli embeddings per la query
    embeddings = OllamaEmbeddings(model=config["embeddings_model"])
    
    # Creazione dell'embedding della domanda
    t0 = time.time()
    query_embedding = embeddings.embed_query(question)
    t1 = time.time()
    print(f"[TIMING] Query embedding: {t1-t0:.2f}s")
    
    # Calcolo vettorizzato di tutte le similarità
    t2 = time.time()
    query_vec = np.array(query_embedding)
    
    # Similarità coseno vettorizzata
    norms = np.linalg.norm(embeddings_matrix, axis=1)
    similarities = np.dot(embeddings_matrix, query_vec) / (norms * np.linalg.norm(query_vec))
    
    t3 = time.time()
    print(f"[TIMING] Computed {len(similarities)} similarities: {t3-t2:.2f}s")
    
    # Retrieving dei documenti più pertinenti
    top_indices = np.argsort(similarities)[::-1][:top_k * 3]
    top_doc_ids = [doc_ids[i] for i in top_indices if similarities[i] >= min_score][:top_k]
    
    # Caricamento dei documenti corrispondenti dal DB
    t4 = time.time()
    results = []
    for doc_id in top_doc_ids:
        doc = collection.find_one({"_id": doc_id}, {"content": 1, "metadata": 1})
        if doc:
            idx = doc_ids.index(doc_id)
            results.append({
                "content": doc["content"],
                "relevance_score": round(float(similarities[idx]), 4),
                "metadata": doc.get("metadata", {})
            })
    
    t5 = time.time()
    print(f"[TIMING] Fetched {len(results)} full documents: {t5-t4:.2f}s")
    print(f"[TIMING] TOTAL (excluding cache): {t5-t0:.2f}s")
    
    return results


# Configurazione del modello LLM e del prompt
model = OllamaLLM(model=config["llm_model"])
template = prompt["template"]
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Caricamento della cache
print("\n" + "="*60)
load_embeddings_cache()
print("="*60)


def format_sources(relevant_articles):
    lines = ["FONTI:"]
    if not relevant_articles:
        lines.append("Nessuna fonte rilevante trovata.")
        return "\n".join(lines)

    # [HACK] Gestione degli URL duplicati mantenendo solo quello con il punteggio di rilevanza più alto
    url_map = {}
    for article in relevant_articles:
        metadata = article.get("metadata", {})
        url = metadata.get("url", "N/A")
        relevance_score = article.get("relevance_score", 0)
        
        if url not in url_map or relevance_score > url_map[url]["relevance_score"]:
            url_map[url] = {"url": url, "relevance_score": relevance_score}
    
    # Ordinamento per punteggio di rilevanza decrescente
    unique_sources = sorted(url_map.values(), key=lambda x: x["relevance_score"], reverse=True)
    
    # Stampa delle fonti con punteggio di rilevanza
    for i, source in enumerate(unique_sources, start=1):
        lines.append(f"{i}. URL: {source['url']}")
        percentage = source['relevance_score'] * 100
        lines.append(f"   Relevance Score: {percentage:.2f}%")
    return "\n".join(lines)


# [TODO: Verificare serva] Pulizia del testo 
def clean_text(text):
    text = text.replace("\\&quot;", '"').replace("&quot;", '"')
    text = text.replace("\\n", " ").replace("\\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 500:
        text = text[:497] + "..."
    return text


# Parsing della risposta del modello
def parse_model_response(result: str):

    # Verdetto
    def infer_verdict(text: str):
        if re.search(r"\bSENZA FONTE\b", text, re.IGNORECASE):
            return "SENZA FONTE"
        if re.search(r"\bVERO\b", text, re.IGNORECASE):
            return "VERO"
        if re.search(r"\bFALSO\b", text, re.IGNORECASE):
            return "FALSO"
        return "SENZA FONTE"

    def parse_json(json_str):
        try:
            parsed = json.loads(json_str)
            verdict = str(parsed.get("verdict", "")).strip().upper()
            explanation = clean_text(str(parsed.get("explanation", "")).strip())
            verdict_valid = verdict in {"VERO", "FALSO", "SENZA FONTE"}
            if not explanation:
                return None, None, False
            return (verdict if verdict_valid else None), explanation, verdict_valid
        except (json.JSONDecodeError, TypeError, ValueError):
            return None, None, False

    # La risposta deve essere un JSON valido con "verdict" e "explanation"    
    verdict, explanation, verdict_valid = parse_json(result)
    if explanation:
        return (verdict if verdict_valid else infer_verdict(explanation)), explanation
    match = re.search(r"\{[\s\S]*\}", result)
    if match:
        verdict, explanation, verdict_valid = parse_json(match.group(0))
        if explanation:
            return (verdict if verdict_valid else infer_verdict(explanation)), explanation
    
    # Estrazione del verdetto
    cleaned = re.sub(r"^```[\s\S]*?\n|```$", "", result).strip()
    verdict_match = re.search(r"\b(VERO|FALSO)\b", cleaned, re.IGNORECASE)
    verdict = verdict_match.group(1).upper() if verdict_match else "FALSO"
    
    if verdict_match:
        cleaned = re.sub(r"\b(VERO|FALSO)\b", "", cleaned, count=1, flags=re.IGNORECASE).strip()
    
    explanation = clean_text(cleaned) if cleaned else "Spiegazione non disponibile."
    return verdict, explanation


# Formattazione degli articoli per il prompt
def format_articles_for_llm(relevant_articles, max_content_per_article: int = 1000):
    formatted = []
    for i, article in enumerate(relevant_articles, start=1):
        metadata = article.get("metadata", {})
        content = article['content']
        
        # Tronca il contenuto se troppo lungo
        if len(content) > max_content_per_article:
            content = content[:max_content_per_article] + "... [troncato]"
        
        info = f"Articolo {i} (Relevance Score: {article['relevance_score']})"
        
        if metadata.get("source"):
            info += f"\nFile: {metadata['source']}"
        
        info += f"\n\nContenuto:\n{content}"
        formatted.append(info)
    
    return "\n\n" + "="*80 + "\n\n".join(formatted)



print("\nBenvenuto al fact-checker!")
while True:
    print("-------------------------------\n ")
    question = input("Che cosa vuoi sapere? [Premi invio per uscire]\n")
    if not question:
        print("Alla prossima!\n")
        break
    
    print("\nAnalisi in corso...\n")
    
    relevant_articles = retrieve_relevant_info(question, min_score=config["similarity_threshold"], top_k=config["max_relevant_articles"])
    formatted_info = format_articles_for_llm(relevant_articles, max_content_per_article=config["article_max_length"])
    
    if not relevant_articles:
        verdict = "SENZA FONTE" # Verdetto fallback
        explanation = "Non sono disponibili fonti rilevanti nel database per verificare questa affermazione."
    else:
        llm_start = time.time()
        result = chain.invoke({"info": formatted_info, "question": question})
        llm_end = time.time()
        print(f"[TIMING] LLM generation: {llm_end-llm_start:.2f}s")
        print(f"[TIMING] Context size: {len(formatted_info)} chars")
        
        verdict, explanation = parse_model_response(result)
    
    if verdict == "SENZA FONTE":
        print(f"Non ci sono fonti rilevanti per verificare questa affermazione.\n")
    else:
        verdict_label = "VERA" if verdict == "VERO" else "FALSA"
        print(f"La seguente affermazione è {verdict_label}!\n")
    print(f"{explanation}\n")
    print(format_sources(relevant_articles))
    print("\n\n")

