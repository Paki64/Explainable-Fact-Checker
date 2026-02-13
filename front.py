import streamlit as st
import requests
import yaml

# Carica configurazioni
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

st.set_page_config(
    page_title="Fact Checker",
    page_icon="🔍",
    layout="centered"
)

st.title("Explainable-Fact-Checker")
st.markdown("Inserisci una notizia per verificarne la veridicità")

# Form per gestire Command+Enter
with st.form("fact_check_form"):
    # Input text area
    news_text = st.text_area(
        " ",
        height=150,
        placeholder="Inserisci qui il testo della notizia..."
    )
    
    # Submit button
    submitted = st.form_submit_button("Verifica", type="primary", use_container_width=True)

# Gestione submit
if submitted:
    if not news_text.strip():
        st.warning("⚠️ Inserisci una notizia prima di verificare")
    else:
        # ✅ Chiama il TUO agent (non Ollama direttamente)
        agent_url = config.get("agent_api_endpoint")
        
        with st.spinner("🔍 Verifico la notizia..."):
            try:
                # Chiamata all'API del tuo agent
                response = requests.post(
                    agent_url,
                    json={"question": news_text},
                    timeout=500  
                )
                response.raise_for_status()
                
                # Parse della risposta strutturata
                result = response.json()
                verdict = result.get("verdict", "").upper()
                explanation = result.get("explanation", "Nessuna spiegazione fornita")
                sources = result.get("sources", [])
                processing_time = result.get("processing_time", 0)
                
                # Display result con banner colorato
                if verdict == "VERO":
                    st.markdown(f"""
                    <div style="background-color: #d4edda; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745;">
                        <h3 style="color: #155724; margin-top: 0;">✓ NOTIZIA VERA</h3>
                        <p style="color: #155724; margin-bottom: 0;">{explanation}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                elif verdict == "FALSO":
                    st.markdown(f"""
                    <div style="background-color: #f8d7da; padding: 20px; border-radius: 10px; border-left: 5px solid #dc3545;">
                        <h3 style="color: #721c24; margin-top: 0;">✗ NOTIZIA FALSA</h3>
                        <p style="color: #721c24; margin-bottom: 0;">{explanation}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                elif verdict == "SENZA FONTE":
                    st.markdown(f"""
                    <div style="background-color: #fff3cd; padding: 20px; border-radius: 10px; border-left: 5px solid #ffc107;">
                        <h3 style="color: #856404; margin-top: 0;">⚠️ SENZA FONTE</h3>
                        <p style="color: #856404; margin-bottom: 0;">{explanation}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info(f"ℹ️ Verdetto: {verdict}")
                    st.write(explanation)
                
                # Mostra fonti se presenti
                if sources:
                    st.markdown("---")
                    st.subheader("📚 Fonti")
                    for i, source in enumerate(sources, 1):
                        url = source.get("url", "N/A")
                        score = source.get("relevance_score", 0) * 100
                        st.markdown(f"{i}. [{url}]({url}) - Rilevanza: **{score:.1f}%**")
                
                # Info timing (opzionale, per debug)
                st.caption(f"⏱️ Tempo di elaborazione: {processing_time:.2f}s")
                
            except requests.exceptions.Timeout:
                st.error("❌ Timeout: l'agent non ha risposto in tempo")
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Impossibile connettersi all'agent su `{agent_url}`")
                st.info("Verifica che l'agent sia in esecuzione e raggiungibile")
            except requests.exceptions.HTTPError as e:
                st.error(f"❌ Errore HTTP {e.response.status_code}")
                if e.response.text:
                    st.code(e.response.text, language="json")
            except Exception as e:
                st.error(f"❌ Errore imprevisto: {str(e)}")

# Sidebar info
with st.sidebar:
    st.header("⚙️ Configurazione")
    
    agent_endpoint = config.get("agent_api_endpoint")
    model_name = config.get("llm_model")
    
    # Estrai la base URL dall'endpoint completo
    agent_base_url = agent_endpoint.rsplit("/api/", 1)[0] if "/api/" in agent_endpoint else agent_endpoint
    
    st.markdown("**Endpoint Agent:**")
    st.code(agent_endpoint, language="text")
    
    st.markdown("**Modello LLM:**")
    st.code(model_name, language="text")
    
    # Health check (opzionale ma utile)
    if st.button("🔌 Test Connessione"):
        try:
            health_response = requests.get(f"{agent_base_url}/api/health", timeout=5)
            if health_response.status_code == 200:
                data = health_response.json()
                st.success(f"✅ Agent attivo - {data.get('num_documents', 0)} documenti in cache")
            else:
                st.error("❌ Agent non risponde correttamente")
        except Exception as e:
            st.error(f"❌ Agent non raggiungibile: {str(e)}")
    
    st.markdown("---")
    st.header("ℹ️ Come Funziona")
    st.markdown("""
    Questo sistema utilizza un **agent RAG** (Retrieval-Augmented Generation) per verificare le notizie:
    
    1. **Retrieval**: cerca documenti rilevanti nel database
    2. **Embedding**: calcola similarità semantica
    3. **LLM Analysis**: il modello analizza fonti e genera verdetto
    
    **Requisiti:**
    - Agent backend in esecuzione
    - Database MongoDB popolato
    - Ollama attivo con il modello configurato
    """)
    
    st.markdown("---")
    st.markdown("Sviluppato da **Giovanni Di Stazio** e **Pasquale Criscuolo** - [GitHub](https://github.com/Paki64/Explainable-Fact-Checker)")
