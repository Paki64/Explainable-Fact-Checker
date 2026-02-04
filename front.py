import streamlit as st
import requests
import os
import yaml
import json

with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

st.set_page_config(
    page_title="Fact Checker",
    page_icon="🔍",
    layout="centered"
)

st.title("Explainable-Fact-Checker")
st.markdown("Inserisci una notizia per verificarne la veridicità")

# Input text area
news_text = st.text_area(
    " ",
    height=150,
    placeholder="Inserisci qui il testo della notizia..."
)

# Submit button
if st.button("Verifica", type="primary", use_container_width=True):
    if not news_text.strip():
        st.warning("Inserisci una notizia prima di verificare")
    else:
        # Get configuration from environment
        ollama_url = config["ollama_api_endpoint"]
        model_name = config["llm_model"]

        with st.spinner("Verifico la notizia..."):
            try:
                # Create prompt for fact-checking
                prompt = f"""Sei un sistema di fact-checking. Analizza la seguente notizia e determina se è vera o falsa. cita le tue fonti se possibile.

Notizia: {news_text}

Rispondi SOLO in formato JSON con questa struttura:
{{
  "verdict": "true" oppure "false",
  "explanation": "spiegazione dettagliata"
}}

Non aggiungere altro testo al di fuori del JSON."""

                # Make API call to Ollama
                response = requests.post(
                    ollama_url,
                    json={
                        "model": model_name,
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=60
                )
                response.raise_for_status()

                # Parse Ollama response
                ollama_result = response.json()
                llm_response = ollama_result.get("response", "")

                # Try to parse JSON from LLM response
                try:
                    # Find JSON in response
                    start_idx = llm_response.find("{")
                    end_idx = llm_response.rfind("}") + 1
                    if start_idx != -1 and end_idx > start_idx:
                        json_str = llm_response[start_idx:end_idx]
                        result = json.loads(json_str)
                    else:
                        raise ValueError("No JSON found in response")

                    verdict = result.get("verdict", "").lower()
                    explanation = result.get("explanation", "Nessuna spiegazione fornita")

                except (json.JSONDecodeError, ValueError):
                    # Fallback: analyze response text
                    st.warning("⚠️ Impossibile parsare la risposta strutturata")
                    st.write("**Risposta del modello:**")
                    st.write(llm_response)
                    verdict = None

                # Display result with colored banner
                if verdict:
                    if verdict in ["true", "vera", "vero", "verified"]:
                        #st.success("✅ NOTIZIA VERIFICATA COME VERA")
                        st.markdown(f"""
                        <div style="background-color: #d4edda; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745;">
                            <h3 style="color: #155724; margin-top: 0;">Verdetto: ✓ VERA</h3>
                            <p style="color: #155724; margin-bottom: 0;">{explanation}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif verdict in ["false", "falsa", "falso", "fake"]:
                        #st.error("❌ NOTIZIA VERIFICATA COME FALSA")
                        st.markdown(f"""
                        <div style="background-color: #f8d7da; padding: 20px; border-radius: 10px; border-left: 5px solid #dc3545;">
                            <h3 style="color: #721c24; margin-top: 0;">Verdetto: ✗ FALSA</h3>
                            <p style="color: #721c24; margin-bottom: 0;">{explanation}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info(f"ℹ️ Verdetto: {verdict}")
                        st.write(explanation)

            except requests.exceptions.Timeout:
                st.error("❌ Timeout: il modello non ha risposto in tempo (aumenta il timeout se necessario)")
            except requests.exceptions.ConnectionError:
                st.error("❌ Errore di connessione: verifica che Ollama sia in esecuzione")
                st.code("ollama serve", language="bash")
            except requests.exceptions.HTTPError as e:
                st.error(f"❌ Errore HTTP: {e.response.status_code}")
            except Exception as e:
                st.error(f"❌ Errore imprevisto: {str(e)}")

# Sidebar info
with st.sidebar:
    st.header("⚙️ Configurazione")

    current_url = config["ollama_api_endpoint"]
    current_model = config["llm_model"]

    st.markdown("**Endpoint API:**")
    st.code(current_url, language="text")

    st.markdown("**Modello in utilizzo:**")
    st.code(current_model, language="text")

    st.markdown("---")
    st.header("ℹ️ Informazioni")
    st.markdown("""
    Questo sistema utilizza Ollama per verificare 
    la veridicità delle notizie.

    **Come usare:**
    1. Inserisci il testo della notizia
    2. Clicca su "Verifica"
    3. Attendi il verdetto
                
    **Nota:**
    - Assicurati che Ollama sia in esecuzione localmente.
    - Configura l'endpoint e il modello nelle variabili d'ambiente.            
    """)

    st.markdown("---")
    st.markdown("Sviluppato da **Giovanni Di Stazio** e **Pasquale Criscuolo** - [GitHub](https://github.com/Paki64/Explainable-Fact-Checker)")