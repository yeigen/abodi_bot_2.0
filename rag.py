import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader

load_dotenv()

openai_api_key = os.environ["OPENAI_API_KEY"]

loader = DirectoryLoader(
    "./markdowns",
    glob="**/*.txt",        
    loader_cls=TextLoader,
    show_progress=True,
)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

vector_store = Chroma(
    collection_name="constitucion",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)
vector_store.add_documents(chunks)