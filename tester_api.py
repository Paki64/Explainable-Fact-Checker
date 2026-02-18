from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import time
import yaml
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from tester import (
    parse_model_response,
    config
)

model = OllamaLLM(model=config["llm_model"])

# Template per verifica online
online_template = """Sei un fact-checker esperto. Il tuo compito è verificare l'affermazione fornita utilizzando le tue conoscenze.

AFFERMAZIONE DA VERIFICARE:
{question}

ISTRUZIONI:
1. Analizza attentamente l'affermazione
2. Valuta la veridicità basandosi sulle tue conoscenze
3. Fornisci una spiegazione dettagliata
4. Se non hai informazioni sufficienti per verificare, indica "SENZA FONTE"

Rispondi in formato JSON:
{{
    "verdict": "VERO" o "FALSO" o "SENZA FONTE",
    "explanation": "Spiegazione dettagliata della tua verifica"
}}

RISPOSTA JSON:"""

prompt_template = ChatPromptTemplate.from_template(online_template)
chain = prompt_template | model

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: verifica configurazione
    print("\n" + "="*60)
    print("[API] Inizializzazione backend online...")
    print(f"[API] Modello LLM: {config['llm_model']}")
    print("[API] Backend pronto! (modalità online - nessun database)")
    print("="*60 + "\n")
    yield
    # Shutdown: cleanup se necessario
    print("\n[API] Shutdown backend...")

app = FastAPI(title="Fact-Checker Online API", version="1.0.0", lifespan=lifespan)

# CORS per permettere richieste da Streamlit o altri frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In produzione: specificare domini precisi
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models per request/response
class FactCheckRequest(BaseModel):
    question: str

class FactCheckResponse(BaseModel):
    verdict: str
    explanation: str
    processing_time: float
    mode: str  # Indica che è modalità "online"

@app.get("/")
async def root():
    return {
        "message": "Fact-Checker Online API",
        "version": "1.0.0",
        "mode": "online (no database)",
        "endpoints": [
            "/api/fact-check",
            "/api/health"
        ]
    }

@app.post("/api/fact-check", response_model=FactCheckResponse)
async def fact_check(request: FactCheckRequest):
    """
    Verifica la veridicità di una notizia usando solo il modello LLM (senza RAG/database)
    """
    start_time = time.time()
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="La domanda non può essere vuota")
    
    try:
        # Verifica diretta con il modello LLM
        result = chain.invoke({"question": request.question})
        verdict, explanation = parse_model_response(result)
        
        processing_time = time.time() - start_time
        
        return FactCheckResponse(
            verdict=verdict,
            explanation=explanation,
            processing_time=processing_time,
            mode="online"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Errore durante l'elaborazione: {str(e)}"
        )

@app.get("/api/health")
async def health():
    """
    Verifica lo stato del backend
    """
    return {
        "status": "ok",
        "mode": "online",
        "database_required": False,
        "model": config["llm_model"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Porta 8001 per non confliggere con agent_api.py
