# agent_api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time

# Importa le funzioni dal tuo agent.py
from agent import (
    load_embeddings_cache,
    retrieve_relevant_info,
    format_articles_for_llm,
    parse_model_response,
    chain,
    config,
    _EMBEDDINGS_CACHE
)

app = FastAPI(title="Fact-Checker API", version="1.0.0")

# CORS per permettere richieste da Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In produzione: ["http://frontend:8501"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models per request/response
class FactCheckRequest(BaseModel):
    question: str

class Source(BaseModel):
    url: str
    relevance_score: float

class FactCheckResponse(BaseModel):
    verdict: str
    explanation: str
    sources: list[Source]
    processing_time: float

# Carica cache all'avvio del server
@app.on_event("startup")
async def startup_event():
    print("\n" + "="*60)
    print("[API] Inizializzazione backend...")
    load_embeddings_cache()
    print("[API] Backend pronto!")
    print("="*60 + "\n")

@app.get("/")
async def root():
    return {
        "message": "Fact-Checker API",
        "version": "1.0.0",
        "endpoints": [
            "/api/fact-check",
            "/api/health"
        ]
    }

@app.post("/api/fact-check", response_model=FactCheckResponse)
async def fact_check(request: FactCheckRequest):
    """
    Verifica la veridicità di una notizia usando RAG + LLM
    """
    start_time = time.time()
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="La domanda non può essere vuota")
    
    try:
        # Retrieval dei documenti rilevanti
        relevant_articles = retrieve_relevant_info(
            request.question, 
            min_score=config["similarity_threshold"], 
            top_k=config["max_relevant_articles"]
        )
        
        # Formatta contesto per LLM
        formatted_info = format_articles_for_llm(
            relevant_articles, 
            max_content_per_article=config["article_max_length"]
        )
        
        # Genera risposta
        if not relevant_articles:
            verdict = "SENZA FONTE"
            explanation = "Non sono disponibili fonti rilevanti nel database per verificare questa affermazione."
        else:
            result = chain.invoke({
                "info": formatted_info, 
                "question": request.question
            })
            verdict, explanation = parse_model_response(result)
        
        # Prepara sources per la risposta
        sources = [
            Source(
                url=article.get("metadata", {}).get("url", "N/A"),
                relevance_score=article["relevance_score"]
            )
            for article in relevant_articles
        ]
        
        processing_time = time.time() - start_time
        
        return FactCheckResponse(
            verdict=verdict,
            explanation=explanation,
            sources=sources,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Errore durante l'elaborazione: {str(e)}"
        )

@app.get("/api/health")
async def health():
    """
    Verifica lo stato del backend e della cache
    """
    cache = _EMBEDDINGS_CACHE
    
    if cache is None:
        return {
            "status": "error",
            "message": "Cache non inizializzata",
            "cache_loaded": False
        }
    
    return {
        "status": "ok",
        "cache_loaded": True,
        "num_documents": len(cache["doc_ids"]),
        "model": config["llm_model"],
        "embeddings_model": config["embeddings_model"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
