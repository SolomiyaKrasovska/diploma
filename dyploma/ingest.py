import os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

FILE_PATH = "data_for_rag.txt"
DB_DIR = "./db"

def create_vector_db():
    if not os.path.exists(FILE_PATH):
        print(f"Помилка: Файл {FILE_PATH} не знайдено! Перевір назву.")
        return

    
    if os.path.exists(DB_DIR):
        print("Видалення старої векторної бази...")
        shutil.rmtree(DB_DIR)

    print("Завантаження тексту...")
    loader = TextLoader(FILE_PATH, encoding="utf-8")
    data = loader.load()

    print("Розбиття тексту на фрагменти...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["---", "\n\n", "\n", " "]
    )
    chunks = text_splitter.split_documents(data)

    print("Створення векторної бази за допомогою nomic-embed-text (це займе кілька секунд)...")
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )

    print("-" * 30)
    print(f"Успіх! Оброблено {len(chunks)} фрагментів.")
    print("Папка 'db' перестворена з правильними ембеддінгами.")
    print("-" * 30)

if __name__ == "__main__":
    create_vector_db()