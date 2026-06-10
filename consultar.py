# consultar.py
# Recibe una pregunta, recupera los chunks más relevantes de ChromaDB y genera una respuesta fundamentada usando Llama

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import chromadb

# 1. CONFIGURAR MODELOS 
# Usamos el mismo modelo de embeddings que en indexar.py
# Esto es obligatorio ya que pregunta y documentos deben vivir en el MISMO
# ESPACIO VECTORIAL para que la similitud coseno tenga sentido.

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# Conectamos con Ollama que esta corriendo localmente.
# request_timeout evita que el script se cuelgue si Ollama tarda.
Settings.llm = Ollama(
    model="llama3.1",
    request_timeout=120.0
)

# 2. CONECTAR CON CHROMADB
# Aqui usa la misma coleccion que creamos en indexar.py.
# Los embeddings ya están ahi (es decir que que no los recalculamos, los leemos)

cliente_chroma = chromadb.PersistentClient(path="chroma_db")
coleccion = cliente_chroma.get_or_create_collection("documentos_empresa")

vector_store = ChromaVectorStore(chroma_collection=coleccion)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 3. RECONSTRUIR EL INDICE
# No reindexamos (por que el índice ya existe en ChromaDB y solo toca decirle a
# LlamaIndex que lo use para buscar)

index = VectorStoreIndex.from_vector_store(
    vector_store,
    storage_context=storage_context,
)

# 4. CREAR EL MOTOR DE CONSULTA
# El query_engine encapsula el pipeline completo:
#   - Convierte la pregunta en embedding
#   - Busca los chunks mas similares en ChromaDB (similarity_top_k)
#   - Construye el prompt aumentado
#   - Lo envía a Ollama y devuelve la respuesta
# similarity_top_k=3 significa que recuperamos los 3 chunks mas relevantes.

query_engine = index.as_query_engine(similarity_top_k=3)

# 5. CONSULTA

pregunta = "¿Puedo teletrabajar los viernes en verano?"

print(f"\nPregunta: {pregunta}")
print("-" * 60)

respuesta = query_engine.query(pregunta)

print(f"Respuesta:\n{respuesta}\n")

# 6. MOSTRAR FUENTES
# Dejamos esto para mostrar la diferencia entre RAG con un LLM normal
# (podemos ver exactamente quu fragmentos uso para responder)

print("Fragmentos utilizados:")
for i, nodo in enumerate(respuesta.source_nodes):
    print(f"\n  [{i+1}] Score de similitud: {nodo.score:.4f}")
    print(f"       Texto: {nodo.text[:200]}...")