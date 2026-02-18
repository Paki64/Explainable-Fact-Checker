import json
import re
import time
import yaml
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

with open("config/prompt.yaml", "r") as file:
    prompt = yaml.safe_load(file)


# Configurazione del modello LLM e del prompt
model = OllamaLLM(model=config["llm_model"])

# Template modificato per verifica online senza RAG
online_template = """Sei un fact-checker esperto. Il tuo compito è verificare l'affermazione fornita utilizzando le tue conoscenze.

AFFERMAZIONE DA VERIFICARE:
{question}

ISTRUZIONI:
1. Analizza attentamente l'affermazione
2. Valuta la veridicità basandoti sulle tue conoscenze
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


# Pulizia del testo 
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


if __name__ == "__main__":
    print("\nBenvenuto al fact-checker online!")
    print("(Modalità: verifica basata sulle conoscenze del modello, senza database)\n")
    
    while True:
        print("-------------------------------\n ")
        question = input("Che cosa vuoi sapere? [Premi invio per uscire]\n")
        if not question:
            print("Alla prossima!\n")
            break
        
        print("\nAnalisi in corso...\n")
        
        # Verifica diretta con il modello LLM senza RAG
        llm_start = time.time()
        result = chain.invoke({"question": question})
        llm_end = time.time()
        print(f"[TIMING] LLM generation: {llm_end-llm_start:.2f}s")
        
        verdict, explanation = parse_model_response(result)
        
        if verdict == "SENZA FONTE":
            print(f"Non ci sono informazioni sufficienti per verificare questa affermazione.\n")
        else:
            verdict_label = "VERA" if verdict == "VERO" else "FALSA"
            print(f"La seguente affermazione è {verdict_label}!\n")
        
        print(f"{explanation}\n")
        print("\nNOTA: Questa verifica si basa sulle conoscenze del modello e non su fonti esterne specifiche.")
        print("\n\n")
