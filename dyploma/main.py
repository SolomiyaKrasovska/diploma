import os
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

MODEL_NAME = "llama3.1" 
DB_DIR = "db"

print("[1/3] Завантаження моделі ембеддінгів...")
embeddings = OllamaEmbeddings(model=MODEL_NAME)

if not os.path.exists(DB_DIR):
    print(f"ПОМИЛКА: Папка {DB_DIR} не знайдена! Спочатку запустіть ingest.py")
    exit()

print("[2/3] Підключення до векторної бази даних ChromaDB...")
vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("[3/3] Налаштування LLM та промпту...")
llm = ChatOllama(model=MODEL_NAME, temperature=0)


template = """Ти — корисний технічний асистент проєкту 'Япомога'. 
Використовуй ТІЛЬКИ наданий контекст для відповіді. 
ВАЖЛИВО: Відповідай ТІЄЮ Ж МОВОЮ, якою користувач поставив запитання.

Контекст:
{context}

Питання: {question}

Відповідь:"""

prompt = ChatPromptTemplate.from_template(template)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


print("\n" + "="*60)
print("СИСТЕМА 'ЯПОМОГА' ГОТОВА ДО РОБОТИ (РЕЖИМ STREAMING)")
print("Для виходу напишіть 'вихід'")
print("="*60 + "\n")

while True:
    user_input = input("Ваше запитання: ").strip()
    
    if not user_input:
        continue
        
    if user_input.lower() in ["вихід", "stop", "exit", "quit"]:
        print("\nДякуємо за використання! Завершення сесії...")
        break

    print("\n[Аналізую базу даних...]")
    print("ВІДПОВІДЬ: ", end="", flush=True)

    try:

        for chunk in rag_chain.stream(user_input):
            print(chunk, end="", flush=True)
        print("\n") 
    except Exception as e:
        print(f"\nВиникла технічна помилка: {e}")
    
    print("-" * 60)