import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader

def main():
    # 1. Cargar documentos
    print("Cargando documentos...")
    loader = DirectoryLoader(
        "./markdowns",
        glob="**/*.txt",        
        loader_cls=TextLoader,
    )
    documents = loader.load()

    # 2. Dividir en chunks
    print("Dividiendo en chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    
    if not chunks:
        print("No se generaron chunks.")
        return

    first_chunk = chunks[0]
    print(f"\nTotal de chunks generados: {len(chunks)}")

    # 3. Inicializar modelo de Embeddings
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    print(f"Inicializando modelo de embeddings: {model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # 4. Generar embedding para el primer chunk
    print("Generando embedding para el primer chunk...")
    vector = embeddings.embed_query(first_chunk.page_content)

    # 5. Mostrar resultados
    print("\n" + "="*50)
    print("DETALLES DEL PRIMER CHUNK")
    print("="*50)
    print(f"Contenido del texto (primeros 300 caracteres):\n{first_chunk.page_content[:300]}...")
    print("-" * 20)
    print(f"Longitud del texto: {len(first_chunk.page_content)} caracteres")
    print(f"Metadatos: {first_chunk.metadata}")
    
    print("\n" + "="*50)
    print("DETALLES DEL VECTOR (EMBEDDING)")
    print("="*50)
    print(f"Dimensiones del vector: {len(vector)}")
    print(f"Primeros 10 valores del vector:\n{vector[:10]}")
    print(f"Tipo de datos: {type(vector)}")

if __name__ == "__main__":
    main()
