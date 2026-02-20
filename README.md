# Explainable-Fact-Checker


## EMBEDDINGS


## MONGO DB


## AGENT

Lo scopo dell' agent è quello di ricevere richieste tramite API dal frontend, elaborare una valutazione tramite l'utilizzo del RAG e comunicare il verdetto sempre tramite API.

**3 Componenti principali** :
-   Server **Ollama** locale con modello LLM.
-   Server **FastAPI** per la gestione delle richieste.
-   Modulo **Agent.py** che gestisce la logica di retrieval e comunicazione col database.

All'avvio dello script di Backend, vengono caricati in RAM tutti gli Embeddings: questo migliora notevolmente le prestazioni del programma, riducendo le interazioni col database.

L'**API** delle richieste è strutturata in questo modo:
-   **GET api/health** strumento di diagnostica, fornisce lo stato attuale del backend e verifica la presenza degli embeddings in RAM.
-   **POST api/fact-check** per la ricezione delle richieste.

All'arrivo di una nuova richiesta, questa viene convertita in Embeddings, viene confrontata con i valori posizionali caricati in RAM e, tramite principio di *similaità cosinusoidale* che si traduce in *similarità semantica* (dato che l'aspetto posizionale e quello semantico sono codipendenti nel database), si recuperano gli articoli più rilevanti per formare la *context window* del modello.

A questo punto il modello ha le informazioni necessarie per valutare la notizia (a meno che non sia stata trovata nessuna fonte rilevante) e risponderà con un *json* formattato in questo modo:
-   **Verdict**: Verdetto principale della valutazione sulla veridicità della notizia, può essere *VERO*, *FALSO* o *SENZA FONTE*, utile principalmente per la visualizzazione nel frontend.
-   **Explanation**: Spiegazione dei motivi che hanno portato al verdetto finale.
-   **Sources**: Riporta le fonti utilizzate per la valutazione, con *url* di riferimento e *relevance score*.
-   **Processing time**: Quanto tempo ha impiegato per la valutazione.

## WEB INTERFACE

Applicazione web interattiva sviluppata con Streamlit per verificare notizie tramite chiamate REST API al backend agent.

**Interfaccia Utente**:

-   **Text area**: Campo per inserire la notizia da verificare.

-   **Pulsante "Verifica"**: Invia richiesta al backend.

-   **Sidebar**: Endpoint agent e modello LLM configurati, Status backend, Numero documenti in cache.

**Flusso Operativo**:

- Utente inserisce notizia

- Validazione input (non vuoto)

- POST request a {agent_api_endpoint} con timeout 500s

- Parsing risposta JSON (verdict, explanation, sources, processing_time)

- Rendering risultato con HTML custom

(Ogni risultato include: Spiegazione dettagliata, Lista fonti con URL cliccabili e percentuale di rilevanza, Tempo di elaborazione)