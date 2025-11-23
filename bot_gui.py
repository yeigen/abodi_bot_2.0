import streamlit as st
import time
import random
import sys
from streamlit.web import cli as stcli
from rag_chain import build_chain, format_sources

if __name__ == "__main__":
    if not st.runtime.exists():
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())

st.set_page_config(page_title="Abodi Bot", page_icon="🤖")

st.title("🤖 Abodi Bot - Asistente Constitucional")

if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def load_chain():
    return build_chain()

if "chain" not in st.session_state:
    with st.spinner("Inicializando el cerebro del bot..."):
        st.session_state.chain = load_chain()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Haz una pregunta sobre la Constitución..."):

    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})


    with st.chat_message("assistant"):
    
        thought_container = st.status("Pensando...", expanded=True)
        
        fake_thoughts = [
            "Analizando la semántica de la pregunta...",
            "Identificando palabras clave...",
            "Consultando la base de datos vectorial (ChromaDB)...",
            "Recuperando fragmentos de la Constitución...",
            "Evaluando relevancia de los documentos...",
            "Sintetizando la respuesta con el modelo LLM...",
            "Verificando fuentes y referencias..."
        ]
        
        # Pick a random subset of thoughts to show sequence
        selected_thoughts = random.sample(fake_thoughts, k=3)
        
        for thought in selected_thoughts:
            thought_container.write(thought)
            time.sleep(random.uniform(0.8, 1.5)) # Random delay
            
        thought_container.update(label="¡Respuesta generada!", state="complete", expanded=False)

        # Real processing
        try:
            response_placeholder = st.empty()
            
            # Invoke chain
            result = st.session_state.chain.invoke({"query": prompt})
            answer = result.get("result", "")
            sources = result.get("source_documents", [])
            formatted_sources = format_sources(sources)
            
            full_response = f"{answer}"
            
            # Simulate typing effect for the final response
            displayed_response = ""
            # Split by words to make it look like typing
            words = full_response.split(" ")
            for i, word in enumerate(words):
                displayed_response += word + " "
                if i % 5 == 0: # Update every 5 words to be smoother/faster
                    response_placeholder.markdown(displayed_response + "▌")
                    time.sleep(0.05)
            
            response_placeholder.markdown(full_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Ocurrió un error: {e}")
