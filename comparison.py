"""
Script per confrontare i verdetti delle due API (RAG vs Online)
"""
import json
import requests
import csv
import time
import os
from typing import List, Dict, Set
import sys
from tqdm import tqdm

# Configurazione API
API_RAG_URL = "http://localhost:8000/api/fact-check"  # API con MongoDB/RAG
API_ONLINE_URL = "http://localhost:8001/api/fact-check"  # API online senza MongoDB

# File di input e output
TESTSET_FILE = "testing/testset.json"
RESULTS_FILE = "testing/results.csv"


def load_testset(filename: str) -> List[str]:
    """
    Carica le affermazioni dal file testset.json
    Formato atteso: una affermazione per riga in formato JSON
    Esempio: {"claim": "La terra è piatta"}
    """
    claims = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Supporta diversi formati: {"claim": "..."} o {"question": "..."} o solo stringa
                    if isinstance(data, dict):
                        claim = data.get('claim') or data.get('question') or data.get('affermazione')
                        if claim:
                            claims.append(claim)
                        else:
                            print(f"[WARNING] Riga {line_num}: nessun campo claim/question trovato")
                    elif isinstance(data, str):
                        claims.append(data)
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Riga {line_num}: errore JSON - {e}")
                    continue
    except FileNotFoundError:
        print(f"[ERROR] File {filename} non trovato!")
        sys.exit(1)
    
    print(f"[INFO] Caricate {len(claims)} affermazioni da {filename}")
    return claims


def load_already_tested_claims(filename: str) -> Set[str]:
    """
    Carica le affermazioni già testate dal CSV esistente
    """
    tested_claims = set()
    if not os.path.exists(filename):
        return tested_claims
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'claim' in row and row['claim']:
                    tested_claims.add(row['claim'])
        
        if tested_claims:
            print(f"[INFO] Trovate {len(tested_claims)} affermazioni già testate nel CSV")
    except Exception as e:
        print(f"[WARNING] Errore nel leggere il CSV esistente: {e}")
    
    return tested_claims


def call_api(url: str, question: str, api_name: str) -> Dict:
    """
    Invia una richiesta all'API e ritorna il verdetto
    """
    try:
        response = requests.post(
            url,
            json={"question": question},
            timeout=120  # Timeout di 2 minuti
        )
        
        if response.status_code == 200:
            data = response.json()
            return {
                "verdict": data.get("verdict", "ERROR"),
                "explanation": data.get("explanation", ""),
                "processing_time": data.get("processing_time", 0),
                "error": None
            }
        else:
            return {
                "verdict": "ERROR",
                "explanation": "",
                "processing_time": 0,
                "error": f"HTTP {response.status_code}"
            }
    except requests.exceptions.Timeout:
        return {
            "verdict": "TIMEOUT",
            "explanation": "",
            "processing_time": 0,
            "error": "Request timeout"
        }
    except Exception as e:
        return {
            "verdict": "ERROR",
            "explanation": "",
            "processing_time": 0,
            "error": str(e)
        }


def append_result_to_csv(result: Dict, is_first: bool = False):
    """
    Appende un risultato al CSV (crea il file se non esiste)
    """
    file_exists = os.path.exists(RESULTS_FILE)
    
    fieldnames = [
        'claim', 
        'verdict_rag', 
        'verdict_online', 
        'match',
        'match_percentage',
        'time_rag',
        'time_online',
        'explanation_rag',
        'explanation_online',
        'error_rag',
        'error_online'
    ]
    
    with open(RESULTS_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Scrivi l'header solo se il file non esiste o è vuoto
        if not file_exists or os.path.getsize(RESULTS_FILE) == 0:
            writer.writeheader()
        
        writer.writerow(result)


def update_match_percentages():
    """
    Aggiorna la percentuale di match per tutte le righe del CSV
    """
    if not os.path.exists(RESULTS_FILE):
        return
    
    # Leggi tutti i risultati
    results = []
    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        results = list(reader)
    
    if not results:
        return
    
    # Calcola la percentuale di match
    total = len(results)
    matches = sum(1 for r in results if r['match'] == '✓')
    match_percentage = f"{(matches / total * 100):.1f}%" if total > 0 else "0.0%"
    
    # Aggiorna tutte le righe con la nuova percentuale
    for r in results:
        r['match_percentage'] = match_percentage
    
    # Riscrivi il file
    fieldnames = [
        'claim', 
        'verdict_rag', 
        'verdict_online', 
        'match',
        'match_percentage',
        'time_rag',
        'time_online',
        'explanation_rag',
        'explanation_online',
        'error_rag',
        'error_online'
    ]
    
    with open(RESULTS_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def test_apis(claims: List[str], already_tested: Set[str]):
    """
    Testa entrambe le API per ogni affermazione e salva i risultati
    """
    # Filtra le affermazioni già testate
    claims_to_test = [c for c in claims if c not in already_tested]
    
    if not claims_to_test:
        print("\n[INFO] Tutte le affermazioni sono già state testate!")
        return
    
    if already_tested:
        print(f"\n[INFO] Skipping {len(already_tested)} affermazioni già testate")
        print(f"[INFO] Rimangono {len(claims_to_test)} affermazioni da testare\n")
    
    # Usa tqdm per la barra di progresso
    with tqdm(total=len(claims_to_test), desc="Testing", unit="claim", ncols=100) as pbar:
        for claim in claims_to_test:
            # Mostra l'affermazione corrente nella barra
            pbar.set_postfix_str(f"{claim[:40]}...")
            
            # Chiamata API RAG (con MongoDB)
            result_rag = call_api(API_RAG_URL, claim, "RAG")
            
            # Pausa breve tra le chiamate
            time.sleep(1)
            
            # Chiamata API Online (senza MongoDB)
            result_online = call_api(API_ONLINE_URL, claim, "Online")
            
            # Prepara il risultato
            match = result_rag["verdict"] == result_online["verdict"]
            result = {
                "claim": claim,
                "verdict_rag": result_rag["verdict"],
                "verdict_online": result_online["verdict"],
                "match": "✓" if match else "✗",
                "match_percentage": "0.0%",  # Sarà aggiornato alla fine
                "time_rag": f"{result_rag['processing_time']:.2f}",
                "time_online": f"{result_online['processing_time']:.2f}",
                "explanation_rag": result_rag["explanation"][:200] if result_rag["explanation"] else "",
                "explanation_online": result_online["explanation"][:200] if result_online["explanation"] else "",
                "error_rag": result_rag["error"] or "",
                "error_online": result_online["error"] or ""
            }
            
            # Salva immediatamente nel CSV
            append_result_to_csv(result)
            
            # Aggiorna la descrizione con il risultato
            status = "✓ Match" if match else "✗ Mismatch"
            pbar.set_description(f"Testing [{status}]")
            
            # Avanza la barra
            pbar.update(1)
            
            # Pausa tra le affermazioni
            time.sleep(2)
    
    # Aggiorna le percentuali di match per tutte le righe
    print(f"\n{'='*80}")
    print(f"Aggiornamento percentuali di match...")
    update_match_percentages()
    print(f"✓ Percentuali aggiornate!")
    
    # Stampa statistiche finali
    print_statistics_from_csv()


def print_statistics_from_csv():
    """
    Legge il CSV e stampa statistiche sui risultati
    """
    if not os.path.exists(RESULTS_FILE):
        print("[WARNING] File CSV non trovato")
        return
    
    results = []
    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        results = list(reader)
    
    if not results:
        print("[WARNING] Nessun risultato nel CSV")
        return
    
    total = len(results)
    matches = sum(1 for r in results if r['match'] == '✓')
    mismatches = total - matches
    
    # Conta verdetti per tipo
    rag_verdicts = {}
    online_verdicts = {}
    
    for r in results:
        rag_v = r['verdict_rag']
        online_v = r['verdict_online']
        
        rag_verdicts[rag_v] = rag_verdicts.get(rag_v, 0) + 1
        online_verdicts[online_v] = online_verdicts.get(online_v, 0) + 1
    
    print(f"\n{'='*80}")
    print(f"STATISTICHE")
    print(f"{'='*80}\n")
    
    print(f"Totale affermazioni testate: {total}")
    print(f"Verdetti concordanti: {matches} ({matches/total*100:.1f}%)")
    print(f"Verdetti discordanti: {mismatches} ({mismatches/total*100:.1f}%)")
    
    print(f"\nAPI RAG (con MongoDB):")
    for verdict, count in sorted(rag_verdicts.items()):
        print(f"  - {verdict}: {count} ({count/total*100:.1f}%)")
    
    print(f"\nAPI Online (senza MongoDB):")
    for verdict, count in sorted(online_verdicts.items()):
        print(f"  - {verdict}: {count} ({count/total*100:.1f}%)")
    
    # Tempi medi
    try:
        avg_time_rag = sum(float(r['time_rag']) for r in results if r['time_rag']) / total
        avg_time_online = sum(float(r['time_online']) for r in results if r['time_online']) / total
        
        print(f"\nTempo medio di risposta:")
        print(f"  - API RAG: {avg_time_rag:.2f}s")
        print(f"  - API Online: {avg_time_online:.2f}s")
        print(f"  - Differenza: {abs(avg_time_rag - avg_time_online):.2f}s")
    except:
        pass
    
    print(f"\n{'='*80}\n")


def check_apis_health():
    """
    Verifica che entrambe le API siano attive
    """
    print("\nVerifica disponibilità API...")
    
    try:
        response = requests.get("http://localhost:8000/api/health", timeout=5)
        if response.status_code == 200:
            print("✓ API RAG (porta 8000) attiva")
        else:
            print("✗ API RAG (porta 8000) non risponde correttamente")
            return False
    except:
        print("✗ API RAG (porta 8000) non raggiungibile")
        print("  Avvia con: python agent_api.py")
        return False
    
    try:
        response = requests.get("http://localhost:8001/api/health", timeout=5)
        if response.status_code == 200:
            print("✓ API Online (porta 8001) attiva")
        else:
            print("✗ API Online (porta 8001) non risponde correttamente")
            return False
    except:
        print("✗ API Online (porta 8001) non raggiungibile")
        print("  Avvia con: python agent_online_api.py")
        return False
    
    return True


if __name__ == "__main__":
    print("\n" + "="*80)
    print("TEST DI CONFRONTO: API RAG vs API ONLINE")
    print("="*80)
    
    # Verifica che le API siano attive
    if not check_apis_health():
        print("\n[ERROR] Una o entrambe le API non sono attive!")
        print("Assicurati di avere entrambe le API in esecuzione prima di lanciare questo test.")
        sys.exit(1)
    
    # Carica il testset
    claims = load_testset(TESTSET_FILE)
    
    if not claims:
        print("\n[ERROR] Nessuna affermazione trovata nel testset!")
        sys.exit(1)
    
    # Carica le affermazioni già testate
    already_tested = load_already_tested_claims(RESULTS_FILE)
    
    # Esegui i test
    test_apis(claims, already_tested)
    
    print("\n✓ Test completato!")
