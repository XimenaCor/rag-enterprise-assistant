# indexar.py
# Este script procesa los documentos de la carpeta /documentos,
# genera embeddings para cada chunk y los almacena en ChromaDB.

import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

# 1. MODELO DE EMBBEDINGS 
# Le decimos a LlamaIndex que modelo usar para generar embeddings.
# La primera vez se descargan los pesos del modelo
# Las siguientes veces se usan desde cache local.

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# Desactivamos el LLM en este script (aqui solo indexamos, no generamos respuestas)
Settings.llm = None
Settings.chunk_size = 256
Settings.chunk_overlap = 20

# 2. CARGAR DOCUMENTOS
# SimpleDirectoryReader lee todos los archivos de la carpeta que se le indica.
# (soporta .txt, .pdf, .docx y otros formatos automaticamente)

print("Cargando documentos...")
documentos = SimpleDirectoryReader("documentos").load_data()
print(f"  {len(documentos)} documento(s) cargado(s)")

# 3. CONFIGURAR CHROMADB
# Creamos el cliente de ChromaDB con persistencia en disco.
# (la carpeta chroma_db se crea automaticamente si no existe)
# Todo lo que indexemos queda guardado ahi entre sesiones.

cliente_chroma = chromadb.PersistentClient(path="chroma_db")

# Nota: Una coleccion en ChromaDB es como una tabla: agrupa vectores relacionados.
# get_or_create garantiza que si ya existe la usamos, si no la creamos.
coleccion = cliente_chroma.get_or_create_collection("documentos_empresa")

# 4. CONECTAR CHROMADB CON LLAMAINDEX
# LlamaIndex necesita un adaptador para hablar con ChromaDB.
# ChromaVectorStore hace de puente entre ambos.

vector_store = ChromaVectorStore(chroma_collection=coleccion)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 5. INDEXAR
# VectorStoreIndex viene a hacer el trabajo pesado:
#   - Divide los documentos en chunks
#   - Llama al modelo de embeddings para cada chunk
#   - Guarda los vectores en ChromaDB

print("Generando embeddings e indexando...")
index = VectorStoreIndex.from_documents(
    documentos,
    storage_context=storage_context,
)

print("Indexación completada.")
print("Los embeddings están guardados en la carpeta chroma_db/")