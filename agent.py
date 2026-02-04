import json
import re
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


# Calcolo della similarità coseno tra due vettori
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# Recupera le informazioni rilevanti dal RAG usando similarità coseno
def retrieve_relevant_info(question: str, min_score: float = 0.35, top_k: int = 20):

    # Connessione al database e inizializzazione degli embeddings
    embeddings = OllamaEmbeddings(model=config["embeddings_model"])
    collection = db_connect()
    
    # Creazione dell'embedding della domanda
    query_embedding = embeddings.embed_query(question)
    
    # Recupero dei documenti dal database
    all_docs = list(collection.find({}, {"content": 1, "embedding": 1, "metadata": 1}))
    if not all_docs:
        print("Nessun documento trovato nel database.")
        return []
    
    # Calcolo della similarità per ogni documento
    similarities = []
    for doc in all_docs:
        if "embedding" in doc:
            relevance_score = cosine_similarity(query_embedding, doc["embedding"])
            similarities.append({
                "content": doc["content"],
                "relevance_score": round(float(relevance_score), 4),
                "metadata": doc.get("metadata", {})
            })
    
    # Ordinamento per similarità decrescente e filtro per punteggio minimo
    similarities.sort(key=lambda x: x["relevance_score"], reverse=True)
    filtered_results = [s for s in similarities if s["relevance_score"] >= min_score]
    return filtered_results[:top_k]


# Configurazione del modello LLM e del prompt
model = OllamaLLM(model=config["llm_model"])
template = prompt["template"]
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


# Formattazione delle fonti per l'output
def format_sources(relevant_articles):
    lines = ["FONTI:"]
    if not relevant_articles:
        lines.append("Nessuna fonte rilevante trovata.")
        return "\n".join(lines)

    # Gestione di URL ridondanti nei chu
    url_map = {}
    for article in relevant_articles:
        metadata = article.get("metadata", {})
        url = metadata.get("url", "N/A")
        relevance_score = article.get("relevance_score", 0)
        
        if url not in url_map or relevance_score > url_map[url]["relevance_score"]:
            url_map[url] = {"url": url, "relevance_score": relevance_score}
    
    # Ordinamento per punteggio di rilevanza decrescente
    unique_sources = sorted(url_map.values(), key=lambda x: x["relevance_score"], reverse=True)
    
    for i, source in enumerate(unique_sources, start=1):
        lines.append(f"{i}. URL: {source['url']}")
        percentage = source['relevance_score'] * 100
        lines.append(f"   Relevance Score: {percentage:.2f}%")
    return "\n".join(lines)


# Pulizia della spiegazione da caratteri indesiderati
def clean_explanation(text):
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
        # SENZA FONTE: non ci sono fonti rilevanti
        if re.search(r"\bSENZA FONTE\b", text, re.IGNORECASE):
            return "SENZA FONTE"
        # VERO: l'affermazione è supportata dalle fonti con punteggio di rilevanza adeguato
        if re.search(r"\bVERO\b", text, re.IGNORECASE):
            return "VERO"
        # FALSO: l'affermazione è contraddetta dalle fonti o non supportata
        if re.search(r"\bFALSO\b", text, re.IGNORECASE):
            return "FALSO"
        return "SENZA FONTE"

    # Verifica la presenza di "verdict" e "explanation" e pulisce l'explanation
    def parse_json(json_str):
        try:
            parsed = json.loads(json_str)
            verdict = str(parsed.get("verdict", "")).strip().upper()
            explanation = clean_explanation(str(parsed.get("explanation", "")).strip())
            verdict_valid = verdict in {"VERO", "FALSO", "SENZA FONTE"}
            if not explanation:
                return None, None, False
            return (verdict if verdict_valid else None), explanation, verdict_valid
        except (json.JSONDecodeError, TypeError, ValueError):
            return None, None, False
    
    # Verifica se la risposta è un JSON valido con i campi richiesti
    verdict, explanation, verdict_valid = parse_json(result)
    if explanation:
        return (verdict if verdict_valid else infer_verdict(explanation)), explanation
    
    # Cerca JSON nel testo
    match = re.search(r"\{[\s\S]*\}", result)
    if match:
        verdict, explanation, verdict_valid = parse_json(match.group(0))
        if explanation:
            return (verdict if verdict_valid else infer_verdict(explanation)), explanation
    
    # Estrazione del verdetto e della spiegazione dal testo libero
    cleaned = re.sub(r"^```[\s\S]*?\n|```$", "", result).strip()
    verdict_match = re.search(r"\b(VERO|FALSO)\b", cleaned, re.IGNORECASE)
    verdict = verdict_match.group(1).upper() if verdict_match else "FALSO"
    
    if verdict_match:
        cleaned = re.sub(r"\b(VERO|FALSO)\b", "", cleaned, count=1, flags=re.IGNORECASE).strip()
    
    explanation = clean_explanation(cleaned) if cleaned else "Spiegazione non disponibile."
    return verdict, explanation


# Formattazione degli articoli per il prompt del LLM
def format_articles_for_llm(relevant_articles):
    formatted = []
    for i, article in enumerate(relevant_articles, start=1):
        metadata = article.get("metadata", {})
        info = f"Articolo {i} (Relevance Score: {article['relevance_score']})"
        
        if metadata.get("source"):
            info += f"\nFile: {metadata['source']}"
        
        info += f"\n\nContenuto:\n{article['content']}"
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
    relevant_articles = retrieve_relevant_info(question)    
    formatted_info = format_articles_for_llm(relevant_articles)
    
    if not relevant_articles:
        verdict = "SENZA FONTE" # Verdetto fallback
        explanation = "Non sono disponibili fonti rilevanti nel database per verificare questa affermazione."
    else:
        result = chain.invoke({"info": formatted_info, "question": question})
        
        verdict, explanation = parse_model_response(result)
    
    if verdict == "SENZA FONTE":
        print(f"Non ci sono fonti rilevanti per verificare questa affermazione.\n")
    else:
        verdict_label = "VERA" if verdict == "VERO" else "FALSA"
        print(f"La seguente affermazione è {verdict_label}!\n")
    print(f"{explanation}\n")
    print(format_sources(relevant_articles))
    print("\n\n")

