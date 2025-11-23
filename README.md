# Abodi Bot - Asistente Constitucional 🤖

Abodi Bot es un asistente inteligente basado en RAG (Retrieval-Augmented Generation) diseñado para responder preguntas sobre la Constitución Política (específicamente la de Colombia, basado en las fuentes). Utiliza modelos de lenguaje avanzados y una base de datos vectorial para proporcionar respuestas precisas y contextualizadas.

## 🚀 Características

- **RAG (Retrieval-Augmented Generation):** Recupera fragmentos relevantes de la constitución para fundamentar sus respuestas.
- **Interfaz Dual:**
  - **CLI (Línea de Comandos):** Para consultas rápidas y registro de métricas.
  - **GUI (Streamlit):** Una interfaz web amigable e interactiva.
- **Base de Datos Vectorial:** Utiliza ChromaDB para búsquedas semánticas eficientes.
- **Modelos de IA:**
  - **Embeddings:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (HuggingFace).
  - **LLM:** `llama-3.1-8b-instant` a través de Groq.

## 📋 Requisitos Previos

Asegúrate de tener instalado Python 3.8 o superior.

Necesitarás configurar las siguientes variables de entorno en un archivo `.env` en la raíz del proyecto:

```env
GROQ_API_KEY=tu_api_key_de_groq
OPENAI_API_KEY=tu_api_key_de_openai # (Si se requiere para scripts de ingestión específicos)
```

## 🛠️ Instalación

1. Clona el repositorio:
   ```bash
   git clone <url-del-repositorio>
   cd abodi_bot
   ```

2. Crea un entorno virtual (opcional pero recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
   *(Nota: Si no tienes un archivo `requirements.txt`, las librerías principales son: `langchain`, `langchain-chroma`, `langchain-huggingface`, `langchain-groq`, `streamlit`, `python-dotenv`, `sentence-transformers`)*

## ▶️ Uso

### 1. Preparación de Datos (Ingestión)

Antes de usar el bot, debes procesar los documentos y crear la base de datos vectorial.

1. Asegúrate de que tus textos (Constitución) estén en la carpeta `markdowns/`.
2. (Opcional) Limpia los textos:
   ```bash
   python clean_txt.py
   ```
3. Genera los embeddings y la base de datos:
   ```bash
   python rag.py
   ```

### 2. Interfaz de Línea de Comandos (CLI)

Para interactuar con el bot desde la terminal y registrar métricas de uso:

```bash
python bot_cli.py
```

### 3. Interfaz Gráfica (GUI)

Para lanzar la aplicación web con Streamlit:

```bash
streamlit run bot_gui.py
```

## 📂 Estructura del Proyecto

- `bot_cli.py`: Interfaz de línea de comandos para el bot.
- `bot_gui.py`: Aplicación web construida con Streamlit.
- `rag.py`: Script para la ingestión de documentos y creación de la base de datos vectorial (ChromaDB).
- `rag_chain.py`: Define la lógica de la cadena RAG, configuración del LLM y el retriever.
- `clean_txt.py`: Utilidad para limpiar y normalizar los archivos de texto de entrada.
- `evaluate_rag.py`: Script para evaluar el rendimiento del sistema RAG.
- `chroma_db/`: Directorio donde se persiste la base de datos vectorial.
- `markdowns/`: Directorio que contiene los documentos fuente (textos de la constitución).
- `rag_metrics.csv`: Archivo donde se registran las métricas de las consultas realizadas vía CLI.

## 📊 Evaluación

El proyecto incluye scripts para evaluar la calidad de las respuestas (`evaluate_rag.py`) y almacena resultados en `evaluation_results.json` y `rag_metrics.csv`.
