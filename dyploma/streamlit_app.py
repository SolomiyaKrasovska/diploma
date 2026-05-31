import os
import datetime
import pandas as pd
import plotly.express as px
import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

MODELL_NAME = "llama3.1"
DB_PAPKA = "db"
EVROPA_CSV = "drs_europe.csv"
PAMYAT_ROZMIR = 6

st.set_page_config(
    page_title="EcoMind AI",
    page_icon="🌍",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp {
    background: linear-gradient(160deg, #0b1f0e 0%, #0d2b1a 35%, #0a2233 65%, #091a2e 100%);
}
header[data-testid="stHeader"] { background: transparent !important; }
.eco-header { text-align: center; padding: 1.5rem 0 1rem; }
.eco-title {
    font-size: 2.5rem; font-weight: 700;
    background: linear-gradient(135deg, #34d399 0%, #22d3ee 50%, #60a5fa 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0 0 0.3rem;
}
.eco-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(52,211,153,0.1); border: 1px solid rgba(52,211,153,0.3);
    border-radius: 24px; padding: 5px 16px; font-size: 0.75rem; color: #6ee7b7;
    text-transform: uppercase; letter-spacing: 0.05em;
}
.dot { width:7px; height:7px; background:#34d399; border-radius:50%;
       animation:blink 2s infinite; display:inline-block; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }
[data-testid="stChatMessage"] {
    border-radius: 16px !important;
    border: 1px solid rgba(52,211,153,0.12) !important;
    margin-bottom: 0.6rem !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg,rgba(34,197,94,0.18),rgba(6,182,212,0.12)) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: rgba(255,255,255,0.03) !important;
}
[data-testid="stChatMessage"] p, 
[data-testid="stChatMessage"] li { 
    color: #ffffff !important; 
    line-height: 1.6; 
}
[data-testid="stChatInput"] {
    background: rgba(255,255,255,0.08) !important;
    border: 1.5px solid rgba(52,211,153,0.5) !important;
    border-radius: 16px !important;
}
[data-testid="stChatInput"] textarea { color: #000000 !important; }
[data-testid="stChatInput"] textarea::placeholder { color: rgba(110,231,183,0.5) !important; }
.stButton button {
    background: linear-gradient(135deg,rgba(34,197,94,0.2),rgba(6,182,212,0.15)) !important;
    border: 1px solid rgba(52,211,153,0.35) !important;
    color: #6ee7b7 !important; border-radius: 12px !important;
}
[data-testid="stSidebar"] {
    background: rgba(11,31,14,0.92) !important;
    border-right: 1px solid rgba(52,211,153,0.15) !important;
}
[data-testid="stSidebar"] * { color: #6ee7b7 !important; }
[data-testid="stExpander"] {
    background: rgba(6,182,212,0.06) !important;
    border: 1px solid rgba(6,182,212,0.15) !important;
    border-radius: 10px !important;
}
hr { border-color: rgba(52,211,153,0.15) !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="eco-header">
    <h1 class="eco-title">🌍 EcoMind AI</h1>
    <div class="eco-badge">
        <span class="dot"></span>
        Інтелектуальний асистент з аналітики депозитного повернення тари
    </div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def zavantazhyty_rag():
    if not os.path.exists(DB_PAPKA):
        return None, None
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = Chroma(persist_directory=DB_PAPKA, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    llm = ChatOllama(model=MODELL_NAME, temperature=0)
    return llm, retriever


@st.cache_data
def zavantazhyty_dani():
    df_evropa = None
    if os.path.exists(EVROPA_CSV):
        df_evropa = pd.read_csv(EVROPA_CSV, encoding="utf-8")
        df_evropa["дата"] = pd.to_datetime(df_evropa["дата"])
    return df_evropa


GRAFIK_STYL = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#ffffff")
)



def chy_analitychne(pytannya):
    pyt = pytannya.lower()
    
    zapyt_slova = [
        "скільки", "топ", "найбільше", "найменше", "статистик", 
        "популярн", "кількість", "відсоток", "аналіз", "графік", 
        "тренд", "динамік", "порівня", "дай дані", "розподіл", "частк"
    ]
    
    return any(slovo in pyt for slovo in zapyt_slova)


def zrobyty_grafik(pytannya, df_evropa):
    pyt = pytannya.lower()
    grafiky = []

    if any(s in pyt for s in ["країна", "рівень", "збір", "відсоток", "порівняй", "європа", "європи", "європі"]):
        if df_evropa is not None:
            rivni = df_evropa.groupby("країна")["рівень_збору_%"].first().reset_index()
            rivni = rivni.sort_values("рівень_збору_%", ascending=True)
            fig = px.bar(rivni, x="рівень_збору_%", y="країна", orientation="h",
                        title="Рівень збору тари по країнах (%)",
                        color="рівень_збору_%", color_continuous_scale="Teal",
                        range_x=[75, 100])
            fig.update_layout(**GRAFIK_STYL)
            grafiky.append(fig)

    if any(s in pyt for s in ["скільки", "кількість", "найбільше", "топ"]):
        if df_evropa is not None:
            po_krainakh = df_evropa.groupby("країна")["кількість_пляшок"].sum().reset_index()
            po_krainakh = po_krainakh.sort_values("кількість_пляшок", ascending=True)
            fig = px.bar(po_krainakh, x="кількість_пляшок", y="країна", orientation="h",
                        title="Загальна кількість пляшок по країнах",
                        color="кількість_пляшок", color_continuous_scale="Greens")
            fig.update_layout(**GRAFIK_STYL)
            grafiky.append(fig)

    if any(s in pyt for s in ["бренд", "марка", "популярний"]):
        if df_evropa is not None:
            brendy = df_evropa.groupby("бренд")["кількість_пляшок"].sum().reset_index()
            brendy = brendy.sort_values("кількість_пляшок", ascending=False).head(10)
            fig = px.bar(brendy, x="бренд", y="кількість_пляшок",
                        title="Топ-10 брендів по кількості пляшок",
                        color="кількість_пляшок", color_continuous_scale="Teal")
            fig.update_layout(**GRAFIK_STYL)
            grafiky.append(fig)

    if any(s in pyt for s in ["рік", "тренд", "динаміка", "зростання"]):
        if df_evropa is not None:
            po_rokakh = df_evropa.groupby("рік")["кількість_пляшок"].sum().reset_index()
            fig = px.line(po_rokakh, x="рік", y="кількість_пляшок",
                         title="Динаміка збору пляшок по роках (Європа)",
                         markers=True, color_discrete_sequence=["#34d399"])
            fig.update_layout(**GRAFIK_STYL)
            grafiky.append(fig)

    if any(s in pyt for s in ["місяць", "сезон", "пора року", "літо", "зима"]):
        if df_evropa is not None:
            sezon = df_evropa.groupby("місяць")["кількість_пляшок"].sum().reset_index()
            nazvy = {1:"Січ",2:"Лют",3:"Бер",4:"Кві",5:"Тра",6:"Чер",
                     7:"Лип",8:"Сер",9:"Вер",10:"Жов",11:"Лис",12:"Гру"}
            sezon["місяць_назва"] = sezon["місяць"].map(nazvy)
            fig = px.bar(sezon, x="місяць_назва", y="кількість_пляшок",
                        title="Сезонність збору по місяцях",
                        color="кількість_пляшок", color_continuous_scale="Teal")
            fig.update_layout(**GRAFIK_STYL)
            grafiky.append(fig)

    if any(s in pyt for s in ["матеріал", "pet", "alu", "пластик", "алюміній"]):
        if df_evropa is not None:
            mat = df_evropa.groupby("матеріал")["кількість_пляшок"].sum().reset_index()
            fig = px.pie(mat, values="кількість_пляшок", names="матеріал",
                        title="Розподіл по типу матеріалу",
                        color_discrete_sequence=["#34d399", "#22d3ee"])
            fig.update_layout(**GRAFIK_STYL)
            grafiky.append(fig)

    if any(s in pyt for s in ["метод", "виплат", "ваучер", "гаманець", "благодійн"]):
        if df_evropa is not None:
            metody = df_evropa.groupby("метод_виплати")["кількість_пляшок"].sum().reset_index()
            fig = px.pie(metody, values="кількість_пляшок", names="метод_виплати",
                        title="Методи виплати депозиту (Європа)",
                        color_discrete_sequence=["#34d399","#22d3ee","#60a5fa","#f59e0b"])
            fig.update_layout(**GRAFIK_STYL)
            grafiky.append(fig)

    return grafiky


def zrobyty_rag_prompt(istoriya, kontekst, pytannya):
    istoriya_tekst = ""
    if istoriya:
        for p in istoriya[-PAMYAT_ROZMIR:]:
            rol = "Користувач" if p["role"] == "user" else "Асистент"
            istoriya_tekst += f"{rol}: {p['content']}\n"
        
    return f"""Ти — помічник EcoMind. Дай відповідь на запитання, використовуючи ТІЛЬКИ текст нижче.

Текст з бази даних:
{kontekst}

Історія розмови:
{istoriya_tekst}

Запитання: {pytannya}
Відповідь:"""


def zrobyty_analitychny_prompt(pytannya, df_evropa):
    stat = ""
    if df_evropa is not None:
      
        po_krainakh = df_evropa.groupby("країна")["кількість_пляшок"].sum().sort_values(ascending=False)
        stat += "Статистика по Європі:\n"
        for k, v in po_krainakh.items():
            riven = df_evropa[df_evropa["країна"]==k]["рівень_збору_%"].iloc[0]
            stat += f"  {k}: {v:,} пляшок, рівень збору {riven}%\n"

    
        brendy = df_evropa.groupby("бренд")["кількість_пляшок"].sum().sort_values(ascending=False).head(5)
        stat += "\nТоп-5 брендів:\n"
        for b, v in brendy.items():
            stat += f"  {b}: {v:,} пляшок\n"

     
        sezon = df_evropa.groupby("місяць")["кількість_пляшок"].sum()
        stat += f"\nНайактивніший місяць (номер): {sezon.idxmax()} ({sezon.max():,} пляшок)\n"


        if "матеріал" in df_evropa.columns:
            mat = df_evropa.groupby("матеріал")["кількість_пляшок"].sum()
            stat += "\nРозподіл за типами матеріалів:\n"
            for m, v in mat.items():
                stat += f"  {m}: {v:,} пляшок\n"

        
        if "метод_виплати" in df_evropa.columns:
            metody = df_evropa.groupby("метод_виплати")["кількість_пляшок"].sum()
            stat += "\nСтатистика методів виплати депозиту:\n"
            for met, v in metody.items():
                stat += f"  {met}: {v:,} пляшок\n"

      
        if "рік" in df_evropa.columns:
            roky = df_evropa.groupby("рік")["кількість_пляшок"].sum()
            stat += "\nДинаміка збору по роках:\n"
            for r, v in roky.items():
                stat += f"  {r} рік: {v:,} пляшок\n"

    return f"""Дай коротку аналітичну відповідь з конкретними цифрами у вигляді звичайного тексту.
Категорично ЗАБОРОНЕНО виводити будь-який Python-код, теги коду, імпорти бібліотек (pandas, matplotlib, plotly) чи інструкції з програмування.
НЕ починай з "Я аналітик", "Звичайно", "Ось відповідь" або будь-яких вступних фраз.
Починай одразу з суті аналізу. Відповідай українською мовою.
Дані стосуються лише європейських країн — НЕ згадуй Україну у статистиці.

Дані для аналізу:
{stat}

Питання: {pytannya}
Відповідь:"""


llm, retriever = zavantazhyty_rag() or (None, None)
df_evropa = zavantazhyty_dani()

if llm is None:
    st.error("⚠️ Папка 'db' не знайдена! Запустіть: python ingest.py")
    st.stop()

if "povidomlennya" not in st.session_state:
    st.session_state.povidomlennya = []


for povidom in st.session_state.povidomlennya:
    with st.chat_message(povidom["role"]):
        st.markdown(povidom["content"])
        
        if povidom.get("grafiky_dani"):
            for gd in povidom["grafiky_dani"]:
                fig = px.bar(**gd) if gd["type"] == "bar" else px.pie(**gd) if gd["type"] == "pie" else px.line(**gd)
                st.plotly_chart(fig, use_container_width=True, theme=None)
        if povidom.get("dzherela"):
            with st.expander("📚 Джерела"):
                for d in povidom["dzherela"]:
                    st.caption(d)


if vvid := st.chat_input("Введіть питання про DRS або аналітику..."):

    if len(vvid.strip()) < 3:
        st.warning("⚠️ Питання занадто коротке!")
        st.stop()

    st.session_state.povidomlennya.append({"role": "user", "content": vvid})
    with st.chat_message("user"):
        st.markdown(vvid)

    with st.chat_message("assistant"):
        dzherela = []
        povna_vidpovid = ""

        if chy_analitychne(vvid):
           
            with st.spinner("📊 Аналізую дані..."):
                try:
                    prompt = zrobyty_analitychny_prompt(vvid, df_evropa)
                    for chunk in llm.stream(prompt):
                        povna_vidpovid += chunk.content
                    st.markdown(povna_vidpovid)
                    dzherela = ["📊 drs_europe.csv"]
                except Exception as error:
                    povna_vidpovid = f"Помилка: {error}"
                    st.error(povna_vidpovid)

            
            grafiky = zrobyty_grafik(vvid, df_evropa)
            for fig in grafiky:
                st.plotly_chart(fig, use_container_width=True, theme=None)

        else:
            with st.spinner("🔍 Аналізую контекст та шукаю в базі..."):
                try:
                   
                    istoriya_dlya_poshuku = ""
                    
                    for p in st.session_state.povidomlennya[-PAMYAT_ROZMIR:-1]:
                        rol = "Користувач" if p["role"] == "user" else "Асистент"
                        istoriya_dlya_poshuku += f"{rol}: {p['content']}\n"
                    
                    poshukovyj_zapyt = vvid
                    
                   
                    if istoriya_dlya_poshuku.strip():
                        perepys_prompt = f"""Історія розмови:
{istoriya_dlya_poshuku}

Поточне запитання: {vvid}

Завдання: Перепиши поточне запитання так, щоб воно стало самостійним (заміни слова "там", "він", "вона", "ця країна" на конкретні назви з історії).
Якщо запитання вже є самостійним, просто поверни його без змін.
НЕ пиши жодних вступних слів, видай ТІЛЬКИ переписане запитання.
Переписане запитання:"""
                        
                        poshukovyj_zapyt = llm.invoke(perepys_prompt).content.strip()
                    
                   
                    docs = retriever.invoke(poshukovyj_zapyt)
                    kontekst = "\n\n".join(doc.page_content for doc in docs)
                    dzherela = [f"📄 {doc.page_content[:80].replace(chr(10),' ')}..." for doc in docs]
                except Exception as e:
                    kontekst = ""
                    dzherela = [f"Помилка пошуку: {e}"]

            prompt = zrobyty_rag_prompt(
                st.session_state.povidomlennya[:-1], kontekst, vvid
            )
            placeholder = st.empty()
            try:
                for chunk in llm.stream(prompt):
                    povna_vidpovid += chunk.content
                    placeholder.markdown(povna_vidpovid + "▌")
                placeholder.markdown(povna_vidpovid)
            except Exception as error:
                povna_vidpovid = f"Помилка: {error}"
                placeholder.error(povna_vidpovid)

        if dzherela:
            with st.expander("📚 Джерела"):
                for d in dzherela:
                    st.caption(d)

    
    st.session_state.povidomlennya.append({
        "role": "assistant",
        "content": povna_vidpovid,
        "dzherela": dzherela
    })


with st.sidebar:
    st.markdown("### 🌍 EcoMind AI")
    st.info(f"**Модель:** {MODELL_NAME}\n\n**Архітектура:** Advanced RAG\n\n**База даних:** ChromaDB\n\n**Пам'ять:** {PAMYAT_ROZMIR} повідомлень\n\n**Аналітика:** drs_europe.csv")
    st.divider()

    kilkist = len(st.session_state.povidomlennya)
    if kilkist > 0:
        st.caption(f"💬 Повідомлень: {kilkist}")
        eksport = f"EcoMind AI — Історія\nДата: {datetime.datetime.now().strftime('%d.%m.%Y %H:%M')}\n{'='*40}\n\n"
        for p in st.session_state.povidomlennya:
            rol = "Ви" if p["role"] == "user" else "EcoMind"
            eksport += f"{rol}:\n{p['content']}\n\n" + "-"*30 + "\n\n"

        st.download_button(
            "💾 Зберегти розмову",
            data=eksport.encode("utf-8"),
            file_name=f"ecomind_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True
        )

    st.divider()
    if st.button("🗑️ Очистити чат", use_container_width=True):
        st.session_state.povidomlennya = []
        st.rerun()
    st.divider()
    st.caption("Дипломна робота\nРозробка засобів генеративного ШІ\nдля аналітики даних")