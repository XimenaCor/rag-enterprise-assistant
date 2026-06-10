# app.py
# Interfaz web del sistema RAG construida con Streamlit.
# Se ejecuta con: streamlit run app.py

import streamlit as st
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import chromadb

# CONFIGURACION DE PAGINA 
# Es lo primero que debe ejecutar Streamlit antes de cualquier otra cosa.

st.set_page_config(
    page_title="Asistente RAG Empresarial",
    #page_icon="🔍",
    layout="centered"
)

# INICIALIZACION
# @st.cache_resource indica a Streamlit que ejecute esta función solo una vez
# y reutilice el resultado en cada interaccion del usuario.
# Sin esto, cargaria el modelo de embeddings en cada pregunta (lo lentea).

@st.cache_resource
def cargar_sistema():
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    Settings.llm = Ollama(
        model="llama3.1",
        request_timeout=120.0
    )
    Settings.chunk_size = 256
    Settings.chunk_overlap = 20

    cliente_chroma = chromadb.PersistentClient(path="chroma_db")
    coleccion = cliente_chroma.get_or_create_collection("documentos_empresa")
    vector_store = ChromaVectorStore(chroma_collection=coleccion)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
    )

    return index.as_query_engine(similarity_top_k=3)

# INTERFAZ

st.title("Asistente RAG Empresarial")
st.caption("Respuestas fundamentadas en documentación interna de la empresa.")
st.divider()

# Cargamos el sistema
query_engine = cargar_sistema()

# Historial de conversación: Streamlit lo mantiene en session_state
# session_state es un diccionario que persiste entre interacciones del usuario.
if "historial" not in st.session_state:
    st.session_state.historial = []

# Mostramos el historial de preguntas y respuestas anteriores
for mensaje in st.session_state.historial:
    with st.chat_message(mensaje["rol"]):
        st.write(mensaje["texto"])

# ENTRADA DEL USUARIO
# st.chat_input crea la caja de texto fija en la parte inferior de la pagina.

pregunta = st.chat_input("Escribe tu pregunta sobre la documentación...")

if pregunta:
    # Se muestra la pregunta del usuario en el chat
    with st.chat_message("user"):
        st.write(pregunta)

    # Guardamos la pregunta en el historial
    st.session_state.historial.append({"rol": "user", "texto": pregunta})

    # Generamos la respuesta con un spinner mientras Ollama trabaja
    with st.chat_message("assistant"):
        with st.spinner("Consultando documentación..."):
            respuesta = query_engine.query(pregunta)

        st.write(str(respuesta))

        # Mostramos las fuentes en un expander desplegable
        with st.expander("📄 Ver fragmentos utilizados"):
            for i, nodo in enumerate(respuesta.source_nodes):
                st.markdown(f"**Fragmento {i+1}** — Score: `{nodo.score:.4f}`")
                st.text(nodo.text[:300])
                st.divider()

    # Guardamos la respuesta en el historial
    st.session_state.historial.append({
        "rol": "assistant",
        "texto": str(respuesta)
    })