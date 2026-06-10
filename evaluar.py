# evaluar.py
# Compara las respuestas de Llama 3.1 con RAG vs sin RAG
# (se usan las mismas preguntas)

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import chromadb

# CONFIGURACION

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
Settings.llm = Ollama(model="llama3.1", request_timeout=120.0)
Settings.chunk_size = 256
Settings.chunk_overlap = 20

# CARGAR INDICE EXISTENTE

cliente_chroma = chromadb.PersistentClient(path="chroma_db")
coleccion = cliente_chroma.get_or_create_collection("documentos_empresa")
vector_store = ChromaVectorStore(chroma_collection=coleccion)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(
    vector_store,
    storage_context=storage_context,
)
query_engine = index.as_query_engine(similarity_top_k=3)

# LLM DIRECTO SIN RAG
# Instanciamos Ollama directamente, sin indice ni contexto.
# (El modelo responde solo con lo que tiene en sus pesos)

llm_directo = Ollama(model="llama3.1", request_timeout=120.0)

# PREGUNTAS DE EVALUACION
# Preguntas especificas sobre el documento

preguntas = [
    "¿Cuántos días de vacaciones tienen los empleados?",
    "¿Puedo teletrabajar los viernes en agosto?",
    "¿Cuánto se reembolsa por kilómetro al usar vehículo propio?",
    "¿Qué ocurre si me pongo enfermo durante las vacaciones?",
]

# COMPARATIVA

separador = "=" * 70

for pregunta in preguntas:
    print(f"\n{separador}")
    print(f"PREGUNTA: {pregunta}")
    print(separador)

    # Respuesta sin RAG -> el LLM responde solo desde sus parámetros
    print("\n SIN RAG (solo LLM):")
    respuesta_sin_rag = llm_directo.complete(pregunta)
    print(respuesta_sin_rag)

    # Respuesta con RAG -> el LLM responde con contexto recuperado
    print("\n CON RAG (LLM + documentos):")
    respuesta_con_rag = query_engine.query(pregunta)
    print(respuesta_con_rag)

    print("\n Fragmentos utilizados por RAG:")
    for i, nodo in enumerate(respuesta_con_rag.source_nodes):
        print(f"  [{i+1}] Score: {nodo.score:.4f} — {nodo.text[:120]}...")

print(f"\n{separador}")
print("Evaluación completada.")