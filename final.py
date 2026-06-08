import chromadb
from openai import OpenAI
import numpy as np
import streamlit as st
import os
import io
from datetime import datetime
import pandas as pd
import db

# ─── CONFIG ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduQuest",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── STYLING ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg: #0a0e1a;
    --surface: #111827;
    --surface2: #1a2235;
    --accent: #6366f1;
    --accent2: #22d3ee;
    --accent3: #f59e0b;
    --success: #10b981;
    --danger: #ef4444;
    --text: #f1f5f9;
    --muted: #64748b;
    --border: #1e293b;
}

html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="block-container"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Force all text to be light */
p, span, div, label, h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span { color: var(--text) !important; }

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background: var(--surface) !important; }

/* Inputs */
.stTextInput input, .stTextArea textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.2) !important;
}
.stTextInput label, .stTextArea label,
.stSelectbox label, .stSlider label,
.stRadio label, .stToggle label { color: var(--text) !important; }

/* Selectbox */
[data-baseweb="select"] > div {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
[data-baseweb="select"] span { color: var(--text) !important; }

/* Buttons */
.stButton button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
.stButton button:hover { background: #4f46e5 !important; }

/* Download button */
.stDownloadButton button {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    border-radius: 8px !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: white !important;
}

/* Metrics */
div[data-testid="metric-container"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
div[data-testid="metric-container"] label,
div[data-testid="metric-container"] [data-testid="stMetricValue"],
div[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: var(--text) !important;
}

/* Expander */
details { background: var(--surface2) !important; border-radius: 8px !important; }
details summary { color: var(--text) !important; }

/* Alerts */
.stAlert { border-radius: 10px !important; }
[data-testid="stAlert"] p { color: inherit !important; }
.stSpinner { color: var(--accent) !important; }

/* Radio */
.stRadio div[role="radiogroup"] label { color: var(--text) !important; }

/* Slider */
.stSlider [data-testid="stTickBar"] { color: var(--muted) !important; }

/* Code blocks */
.stCode { background: var(--surface) !important; }

.card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 0.75rem;
}
.card-accent { border-left: 3px solid var(--accent); }
.xp-badge {
    display: inline-block;
    background: linear-gradient(135deg, var(--accent3), #f97316);
    color: #000;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.8rem;
}
.rank-1 { color: #fbbf24; font-weight: 700; }
.rank-2 { color: #94a3b8; font-weight: 700; }
.rank-3 { color: #b45309; font-weight: 700; }
.level-badge {
    background: var(--surface);
    border: 1px solid var(--accent);
    color: var(--accent);
    padding: 2px 12px;
    border-radius: 999px;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
}
.session-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s;
}
.session-card:hover { border-color: var(--accent); }
</style>
""", unsafe_allow_html=True)


# ─── DB INIT ──────────────────────────────────────────────────────────────────
@st.cache_resource
def _init_db():
    db.init_db()
    return True

_init_db()


# ─── CLIENTS (OpenAI + ChromaDB) ──────────────────────────────────────────────
@st.cache_resource
def get_clients():
    api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    client = OpenAI(api_key=api_key)

    chroma_api_key = st.secrets.get("CHROMA_API_KEY", os.environ.get("CHROMA_API_KEY", ""))
    if chroma_api_key:
        chroma = chromadb.CloudClient(
            tenant=st.secrets.get("CHROMA_TENANT", os.environ.get("CHROMA_TENANT", "")),
            database=st.secrets.get("CHROMA_DATABASE", os.environ.get("CHROMA_DATABASE", "")),
            api_key=chroma_api_key
        )
    else:
        chroma = chromadb.PersistentClient(path="./chroma_db")

    collection = chroma.get_or_create_collection(
        name="school_booksv2",
        metadata={"hnsw:space": "cosine"}
    )
    return client, collection


SOURCE_MAP = {
    "almaty_7_class": "Информатика, 7 класс — Алматыкітап баспасы, 2021",
}

GRADE_BOOKS = {
    5:  ["arman_pv_5", "arman_pv_5_kaz"],
    6:  ["6_class_rus", "6_class_kz"],
    7:  ["almaty_7", "almaty_7_class", "almaty_7_kaz", "arman_pv_7_watermark"],
    8:  ["inf_8_okulik", "arman_pv_8", "arman_pv_8_kaz"],
    9:  ["arman_pv_9_watermark", "arman_pv_9_kaz"],
    10: ["arman_pv_10", "arman_pv_10_kaz"],
    11: ["arman_pv_11", "arman_pv_11_kaz"],
}


# ─── RAG ──────────────────────────────────────────────────────────────────────
def embed(text, client):
    r = client.embeddings.create(model="text-embedding-3-large", input=text)
    v = np.array(r.data[0].embedding)
    return (v / np.linalg.norm(v)).tolist()


def ask(query, collection, client, top_k=5, grade=None):
    q_emb = embed(query, client)
    kwargs = dict(query_embeddings=[q_emb], n_results=top_k,
                  include=["documents", "metadatas", "distances"])
    if grade and grade in GRADE_BOOKS:
        kwargs["where"] = {"book": {"$in": GRADE_BOOKS[grade]}}
    res = collection.query(**kwargs)
    results = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        label = meta.get("label") or SOURCE_MAP.get(meta["book"], meta["book"])
        results.append({"text": doc, "book": label, "page": meta["page"], "dist": float(dist)})
    return results


def get_context_text(question, collection, client, grade=None):
    contexts = ask(question, collection, client, top_k=5, grade=grade)
    texts = [c["text"] for c in contexts]
    sources = list({f'{c["book"]}, стр. {c["page"]}' for c in contexts})
    return "\n\n".join(texts), "; ".join(sources)


def keyword_score(context, answer):
    cw = set(context.lower().split())
    aw = set(answer.lower().split())
    return len(cw & aw) / max(len(cw), 1)


EVAL_PROMPT = {
    "ru": """Ты система проверки знаний школьников. Отвечай ТОЛЬКО на русском языке.

ПРАВИЛА:
- Используй ТОЛЬКО предоставленный учебный материал
- НЕ добавляй свои примеры
- Будь строгим но справедливым

ЗАДАЧА:
1. Сравни ответ студента с учебным материалом
2. Поставь оценку (0-100)
3. Покажи что ОТСУТСТВУЕТ (только из текста)
4. Покажи что ВЕРНО (если есть)
5. Дай ПРАВИЛЬНЫЙ ОТВЕТ строго по учебнику
6. Дай краткую обратную связь

ОЦЕНИВАНИЕ:
- Очень краткий или частичный ответ → 20–40
- Ответ без свойств и деталей → максимум 50
- Полный ответ (определение + детали) → 80+

Вопрос: {question}
Ответ студента: {student_answer}
Учебный материал: {ctx}
Источники: {source_text}

Ответ строго в этом формате:
Score: [число]
Отсутствует: [пункты]
Верно: [пункты]
Правильный ответ: [полный ответ строго по тексту учебника]
Обратная связь: [1-2 предложения]
Источник: {source_text}""",

    "kz": """Сен мектеп оқушыларының білімін тексеру жүйесісің. ТЕК қазақ тілінде жауап бер.

ЕРЕЖЕЛЕР:
- ТЕК берілген оқулық материалды пайдалан
- Өз мысалдарыңды ҚОСПА
- Қатаң бол, бірақ әділ бол

ТАПСЫРМА:
1. Оқушының жауабын оқулық материалымен салыстыр
2. Баға қой (0-100)
3. ЖЕТІСПЕЙТІНІН көрсет (тек мәтіннен)
4. ДҰРЫСЫН көрсет (бар болса)
5. Оқулық бойынша ДҰРЫС ЖАУАП бер
6. Қысқа кері байланыс бер

БАҒАЛАУ:
- Өте қысқа немесе жартылай жауап → 20–40
- Қасиеттері мен егжей-тегжейлері жоқ жауап → ең көбі 50
- Толық жауап (анықтама + егжей-тегжей) → 80+

Сұрақ: {question}
Оқушы жауабы: {student_answer}
Оқулық материалы: {ctx}
Дереккөздер: {source_text}

Жауап осы форматта:
Score: [сан]
Жетіспейді: [тармақтар]
Дұрыс: [тармақтар]
Дұрыс жауап: [оқулық бойынша толық жауап]
Кері байланыс: [1-2 сөйлем]
Дереккөз: {source_text}"""
}


def evaluate_answer(question, student_answer, collection, client, grade=None, lang="ru"):
    context_text, source_text = get_context_text(question, collection, client, grade=grade)
    ctx = context_text[:3000]
    prompt = EVAL_PROMPT.get(lang, EVAL_PROMPT["ru"]).format(
        question=question,
        student_answer=student_answer,
        ctx=ctx,
        source_text=source_text
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.choices[0].message.content

    score = 50
    for line in raw.split("\n"):
        if line.strip().startswith("Score:"):
            try:
                score = int(''.join(filter(str.isdigit,
                            line.split(":", 1)[1].strip().split()[0])))
            except Exception:
                pass

    return {"feedback": raw, "score": score, "source": source_text}


def answer_with_llm(query, collection, client, top_k=5, grade=None):
    contexts = ask(query, collection, client, top_k, grade=grade)
    if not contexts:
        return "Ответ не найден в базе знаний."
    context_text = "\n\n".join([f"[{i+1}] {c['text']}" for i, c in enumerate(contexts)])
    prompt = f"""Ты учебный ассистент.
Отвечай ТОЛЬКО на основе предоставленных фрагментов учебника.
Указывай номера источников в скобках.

Фрагменты:
{context_text}

Вопрос: {query}

Ответ:"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()


# ─── XP & LEVELS ──────────────────────────────────────────────────────────────
XP_THRESHOLDS = [0, 50, 150, 300, 500, 750, 1000, 1500, 2000, 3000]
XP_NAMES = ["Newcomer", "Learner", "Student", "Scholar",
            "Expert", "Master", "Champion", "Legend", "Guru", "Sage"]


def score_to_xp(score: int) -> int:
    return max(5, int(score * 1.5))


def xp_to_level(xp: int):
    for i in range(len(XP_THRESHOLDS) - 1, -1, -1):
        if xp >= XP_THRESHOLDS[i]:
            next_xp = XP_THRESHOLDS[i + 1] if i + 1 < len(XP_THRESHOLDS) else XP_THRESHOLDS[-1]
            return XP_NAMES[i], i + 1, XP_THRESHOLDS[i], next_xp
    return "Newcomer", 1, 0, 50


def xp_progress_bar(xp: int) -> int:
    _, _, curr, nxt = xp_to_level(xp)
    if nxt == curr:
        return 100
    return int((xp - curr) / (nxt - curr) * 100)


def rank_medal(rank: int) -> str:
    return ["🥇", "🥈", "🥉"][rank - 1] if rank <= 3 else f"#{rank}"


# ─── EXCEL EXPORT ─────────────────────────────────────────────────────────────
def build_excel(session_id: int, session_title: str) -> bytes:
    rows = db.get_session_export(session_id)
    lb = db.get_session_leaderboard(session_id)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_answers = pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()
        df_answers.to_excel(writer, sheet_name="Ответы", index=False)
        if lb:
            pd.DataFrame([dict(r) for r in lb]).to_excel(writer, sheet_name="Рейтинг", index=False)
    return buf.getvalue()


# ─── LEADERBOARD WIDGET ───────────────────────────────────────────────────────
def render_leaderboard(rows, current_name=None, xp_field="session_xp"):
    if not rows:
        st.info("Нет данных")
        return
    html = ""
    for i, row in enumerate(rows):
        name = row["name"]
        xp = row[xp_field]
        medal = rank_medal(i + 1)
        is_me = (name == current_name)
        ln, lnum, _, _ = xp_to_level(xp)
        highlight = "border-left: 3px solid #6366f1;" if is_me else ""
        me_tag = "<em style='color:#64748b'> (вы)</em>" if is_me else ""
        html += f"""
        <div style='background:#1a2235; border:1px solid #1e293b; border-radius:12px;
                    padding:0.75rem 1rem; margin-bottom:0.5rem; {highlight}'>
            <span style='font-size:1.2rem'>{medal}</span>
            <strong style='margin-left:0.5rem'>{name}</strong>{me_tag}
            <span style='display:inline-block; background:linear-gradient(135deg,#f59e0b,#f97316);
                  color:#000; font-weight:700; font-family:Space Mono,monospace;
                  padding:2px 10px; border-radius:999px; font-size:0.8rem;
                  margin-left:0.5rem'>⚡ {xp} XP</span>
            <span style='background:#111827; border:1px solid #6366f1; color:#6366f1;
                  padding:2px 12px; border-radius:999px; font-family:Space Mono,monospace;
                  font-size:0.75rem; margin-left:0.5rem'>Lv.{lnum} {ln}</span>
        </div>"""
    st.markdown(html, unsafe_allow_html=True)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem;'>
        <span style='font-family: Space Mono, monospace; font-size: 2rem; font-weight:700;
        background: linear-gradient(135deg, #6366f1, #22d3ee);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>EduQuest</span>
        <span style='color:#64748b; font-size:0.9rem; margin-left:0.5rem;'>/ Collaborative Learning</span>
    </div>
    """, unsafe_allow_html=True)

    for key in ("role", "user_id", "username", "grade", "session_id", "chat_messages"):
        if key not in st.session_state:
            st.session_state[key] = None

    if st.session_state.role is None:
        _login_screen()
    elif st.session_state.role == "teacher":
        teacher_view()
    elif st.session_state.role == "student":
        student_view()


# ─── LOGIN ────────────────────────────────────────────────────────────────────
def _login_screen():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        role = st.radio("Я...", ["👨‍🏫 Учитель", "👨‍🎓 Ученик"], horizontal=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if role == "👨‍🏫 Учитель":
            st.markdown("### Вход для учителя")
            name = st.text_input("Имя")
            password = st.text_input("Пароль", type="password")
            if st.button("Войти как учитель", use_container_width=True):
                teacher = db.verify_teacher(name.strip(), password)
                if teacher:
                    st.session_state.role = "teacher"
                    st.session_state.user_id = teacher["id"]
                    st.session_state.username = teacher["name"]
                    st.rerun()
                else:
                    st.error("Неверное имя или пароль")
        else:
            st.markdown("### Вход для ученика")
            name = st.text_input("Имя")
            password = st.text_input("Пароль", type="password")
            if st.button("Войти", use_container_width=True):
                student = db.verify_student(name.strip(), password)
                if student:
                    st.session_state.role = "student"
                    st.session_state.user_id = student["id"]
                    st.session_state.username = student["name"]
                    st.session_state.grade = student["grade"]
                    st.session_state.session_id = None
                    st.session_state.chat_messages = []
                    st.rerun()
                else:
                    st.error("Неверное имя или пароль")


# ─── TEACHER VIEW ─────────────────────────────────────────────────────────────
def teacher_view():
    client, collection = get_clients()

    col_title, col_logout = st.columns([5, 1])
    with col_title:
        st.markdown(f"### 👨‍🏫 Панель учителя — *{st.session_state.username}*")
    with col_logout:
        if st.button("Выйти"):
            for k in ("role", "user_id", "username", "grade", "session_id"):
                st.session_state[k] = None
            st.rerun()

    tab1, tab2, tab3 = st.tabs(["🆕 Создать сессию", "📋 Сессии", "📊 Аналитика"])

    # ── СОЗДАТЬ ──────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("#### Новая сессия")
        col1, col2 = st.columns([3, 2])
        with col1:
            title = st.text_input("Название сессии", placeholder="Глава 5 — Повторение")
            gcol, lcol = st.columns(2)
            with gcol:
                grade = st.selectbox("Класс", options=sorted(GRADE_BOOKS.keys()),
                                     format_func=lambda g: f"{g} класс")
            with lcol:
                lang = st.selectbox("Язык", options=["ru", "kz"],
                                    format_func=lambda l: "Русский" if l == "ru" else "Қазақша")
            question = st.text_area("Вопрос для учеников", height=120,
                                    placeholder="Что такое оперативная память и как она работает?")
            time_limit = st.slider("Лимит времени (мин)", 2, 60, 10)

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🚀 Создать сессию", use_container_width=True):
            if not question.strip():
                st.error("Введите вопрос")
            else:
                row = db.create_session(
                    title=title or "Квиз",
                    question=question.strip(),
                    grade=grade,
                    lang=lang,
                    time_limit=time_limit,
                    chat_enabled=False,
                    teacher_id=st.session_state.user_id
                )
                st.success("Сессия создана!")
                st.markdown(f"""
                <div style='background:#1a2235; border:2px solid #6366f1; border-radius:12px;
                     padding:1.5rem; text-align:center; margin-top:1rem;'>
                    <div style='color:#64748b; font-size:0.9rem; margin-bottom:0.5rem;'>
                        Код сессии для учеников:
                    </div>
                    <div style='font-family: Space Mono, monospace; font-size:3rem; font-weight:700;
                         color:#22d3ee; letter-spacing:0.3em;'>{row["code"]}</div>
                    <div style='color:#64748b; font-size:0.8rem; margin-top:0.5rem;'>
                        {grade} класс · {time_limit} мин · {title or "Квиз"}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── СЕССИИ ───────────────────────────────────────────────────────────────
    with tab2:
        sessions = db.get_all_sessions()
        if not sessions:
            st.info("Сессий пока нет.")
        else:
            for sess in sessions:
                sid = sess["id"]
                status_color = "#10b981" if sess["status"] == "active" else "#64748b"
                with st.expander(
                    f"**{sess['title']}** · {sess['grade']} кл · "
                    f"Код: `{sess['code']}` · {sess['student_count']} уч."
                ):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.markdown(f"**Вопрос:** {sess['question']}")
                        st.markdown(
                            f"**Статус:** <span style='color:{status_color}'>"
                            f"{sess['status'].upper()}</span>",
                            unsafe_allow_html=True
                        )
                        st.caption(f"Создано: {sess['created_at']}")
                    with col2:
                        if sess["status"] == "active":
                            if st.button("⏹ Закрыть", key=f"close_{sid}"):
                                db.set_session_status(sid, "closed")
                                st.rerun()
                        else:
                            if st.button("🔄 Открыть", key=f"open_{sid}"):
                                db.set_session_status(sid, "active")
                                st.rerun()
                    with col3:
                        xl = build_excel(sid, sess["title"])
                        st.download_button(
                            "📥 Excel",
                            data=xl,
                            file_name=f"session_{sess['code']}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"dl_{sid}"
                        )

                    # Session leaderboard
                    lb = db.get_session_leaderboard(sid)
                    if lb:
                        st.markdown("##### 🏆 Рейтинг сессии")
                        render_leaderboard(lb, xp_field="session_xp")

    # ── АНАЛИТИКА ────────────────────────────────────────────────────────────
    with tab3:
        sessions = db.get_all_sessions()
        total_sessions = len(sessions)
        total_students = sum(s["student_count"] for s in sessions)

        col1, col2, col3 = st.columns(3)
        col1.metric("Всего сессий", total_sessions)
        col2.metric("Всего участников", total_students)
        col3.metric("Активных", sum(1 for s in sessions if s["status"] == "active"))


# ─── STUDENT VIEW ─────────────────────────────────────────────────────────────
def student_view():
    client, collection = get_clients()

    # Refresh student info (XP may have changed)
    info = db.get_student_info(st.session_state.user_id)
    xp = info["total_xp"]
    grade = info["grade"]
    level_name, level_num, _, _ = xp_to_level(xp)
    progress_pct = xp_progress_bar(xp)

    # Header
    col_title, col_xp, col_logout = st.columns([3, 2, 1])
    with col_title:
        if st.session_state.session_id:
            if st.button("← Назад к сессиям"):
                st.session_state.session_id = None
                st.session_state.chat_messages = []
                st.rerun()
        st.markdown(f"### 👋 *{st.session_state.username}* — {grade} класс")
    with col_xp:
        st.markdown(f"""
        <div style='text-align:right; padding-top:0.5rem;'>
            <span class='xp-badge'>⚡ {xp} XP</span>
            <span class='level-badge' style='margin-left:0.5rem'>Lv.{level_num} {level_name}</span>
        </div>
        <div style='background:#1e293b; border-radius:999px; height:6px; margin-top:6px;'>
            <div style='background:linear-gradient(90deg,#6366f1,#22d3ee);
                 width:{progress_pct}%; height:100%; border-radius:999px;'></div>
        </div>
        """, unsafe_allow_html=True)
    with col_logout:
        if st.button("Выйти"):
            for k in ("role", "user_id", "username", "grade", "session_id", "chat_messages"):
                st.session_state[k] = None
            st.rerun()

    st.markdown("---")

    # ── HOME SCREEN (no session selected) ────────────────────────────────────
    if not st.session_state.session_id:
        home_tab1, home_tab2, home_tab3 = st.tabs(
            ["📚 Сессии", f"🌍 Рейтинг {grade} класса", "💬 AI Чат"]
        )

        with home_tab1:
            st.markdown(f"#### Открытые сессии для {grade} класса")
            sessions = db.get_sessions_for_grade(grade)
            if not sessions:
                st.info("Нет активных сессий. Ожидайте, учитель создаст сессию.")
            else:
                for sess in sessions:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"""
                        <div style='background:#1a2235; border:1px solid #1e293b; border-radius:12px;
                             padding:1.25rem; margin-bottom:0.75rem;'>
                            <div style='font-size:1.1rem; font-weight:600; color:#f1f5f9;'>{sess['title']}</div>
                            <div style='color:#64748b; font-size:0.85rem; margin-top:0.25rem;'>
                                Учитель: {sess['teacher_name']} &nbsp;·&nbsp;
                                Осталось: {sess['mins_left']} мин
                            </div>
                            <div style='margin-top:0.5rem; color:#94a3b8;'>{sess['question'][:120]}...</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown("<br><br><br>", unsafe_allow_html=True)
                        if st.button("Войти →", key=f"join_{sess['id']}"):
                            st.session_state.session_id = sess["id"]
                            st.session_state.chat_messages = []
                            st.rerun()

        with home_tab2:
            st.markdown(f"#### 🌍 Рейтинг {grade} класса — всё время")
            st.caption("Суммарный XP за все сессии")
            class_lb = db.get_class_leaderboard(grade)
            render_leaderboard(class_lb, current_name=st.session_state.username, xp_field="total_xp")
            if st.button("🔄 Обновить"):
                st.rerun()

        with home_tab3:
            st.markdown("#### 💬 AI Чат по учебнику")
            st.caption(f"Вопросы по информатике {grade} класса")
            if not st.session_state.chat_messages:
                st.session_state.chat_messages = []
            for msg in st.session_state.chat_messages:
                st.chat_message(msg["role"]).write(msg["content"])
            user_input = st.chat_input("Задайте вопрос по учебнику...")
            if user_input:
                st.session_state.chat_messages.append({"role": "user", "content": user_input})
                st.chat_message("user").write(user_input)
                with st.spinner("Ищу в учебнике..."):
                    answer = answer_with_llm(user_input, collection, client, grade=grade)
                st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                st.chat_message("assistant").write(answer)
        return

    # ── INSIDE SESSION ────────────────────────────────────────────────────────
    sess = db.get_session(st.session_state.session_id)
    if not sess:
        st.error("Сессия не найдена.")
        st.session_state.session_id = None
        st.rerun()
        return

    if sess["status"] != "active":
        st.warning("⏹ Сессия закрыта учителем или время истекло.")

    tab1, tab2 = st.tabs(["📝 Ответить", "🏆 Рейтинг сессии"])

    # ── TAB 1: ANSWER ─────────────────────────────────────────────────────────
    with tab1:
        st.markdown(f"""
        <div style='background:#1a2235; border-left:3px solid #6366f1; border-radius:12px;
             padding:1.25rem; margin-bottom:1.5rem;'>
            <div style='color:#64748b; font-size:0.8rem; text-transform:uppercase;
                 letter-spacing:0.1em; margin-bottom:0.5rem;'>📌 Вопрос</div>
            <div style='font-size:1.1rem; font-weight:500; color:#f1f5f9;'>{sess['question']}</div>
        </div>
        """, unsafe_allow_html=True)

        existing = db.get_student_answer(st.session_state.user_id, sess["id"])

        if existing:
            st.success("✅ Вы уже ответили на этот вопрос.")
            score_color = "#10b981" if existing["score"] >= 70 else "#f59e0b" if existing["score"] >= 40 else "#ef4444"
            st.markdown(f"""
            <div style='display:flex; gap:1rem; margin:1rem 0;'>
                <div class='card' style='flex:1; text-align:center;'>
                    <div style='font-size:2.5rem; font-weight:700; color:{score_color};
                         font-family:Space Mono,monospace;'>{existing['score']}</div>
                    <div style='color:#64748b; font-size:0.85rem;'>/ 100 Score</div>
                </div>
                <div class='card' style='flex:1; text-align:center;'>
                    <div style='font-size:2.5rem; font-weight:700; color:#f59e0b;
                         font-family:Space Mono,monospace;'>+{existing['xp_earned']}</div>
                    <div style='color:#64748b; font-size:0.85rem;'>XP Earned</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("📋 Подробная обратная связь"):
                st.code(existing["feedback"], language=None)

        elif sess["status"] == "active":
            student_answer = st.text_area(
                "Ваш ответ", height=150,
                placeholder="Напишите ответ здесь...",
                key="student_answer_input"
            )
            if st.button("✅ Отправить ответ", use_container_width=True):
                if not student_answer.strip():
                    st.warning("Напишите ответ перед отправкой.")
                else:
                    with st.spinner("ИИ оценивает ваш ответ..."):
                        result = evaluate_answer(
                            sess["question"], student_answer, collection, client,
                            grade=sess.get("grade"), lang=sess.get("lang", "ru")
                        )

                    score = result["score"]
                    xp_earned = score_to_xp(score)
                    db.submit_answer(
                        student_id=st.session_state.user_id,
                        session_id=sess["id"],
                        answer_text=student_answer,
                        score=score,
                        xp_earned=xp_earned,
                        feedback=result["feedback"]
                    )

                    score_color = "#10b981" if score >= 70 else "#f59e0b" if score >= 40 else "#ef4444"
                    st.markdown(f"""
                    <div style='display:flex; gap:1rem; margin:1rem 0;'>
                        <div class='card' style='flex:1; text-align:center;'>
                            <div style='font-size:2.5rem; font-weight:700; color:{score_color};
                                 font-family:Space Mono,monospace;'>{score}</div>
                            <div style='color:#64748b; font-size:0.85rem;'>/ 100 Score</div>
                        </div>
                        <div class='card' style='flex:1; text-align:center;'>
                            <div style='font-size:2.5rem; font-weight:700; color:#f59e0b;
                                 font-family:Space Mono,monospace;'>+{xp_earned}</div>
                            <div style='color:#64748b; font-size:0.85rem;'>XP Earned</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    with st.expander("📋 Подробная обратная связь", expanded=True):
                        st.code(result["feedback"], language=None)
                    st.rerun()
        else:
            st.info("Сессия закрыта. Отправка ответов недоступна.")

    # ── TAB 2: SESSION LEADERBOARD ────────────────────────────────────────────
    with tab2:
        st.markdown(f"#### 🏆 {sess['title']} — Рейтинг сессии")
        st.caption("XP заработанные в этой сессии")
        lb = db.get_session_leaderboard(sess["id"])
        render_leaderboard(lb, current_name=st.session_state.username, xp_field="session_xp")
        if st.button("🔄 Обновить", key="refresh_session_lb"):
            st.rerun()


if __name__ == "__main__":
    main()
