import fitz
import chromadb
from openai import OpenAI
import numpy as np
import streamlit as st
import json
import time
import os
from datetime import datetime
import hashlib

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

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background: var(--surface) !important; }

.stTextInput input, .stTextArea textarea, .stSelectbox select {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.2) !important;
}

.stButton button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton button:hover {
    background: #4f46e5 !important;
    transform: translateY(-1px) !important;
}

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
    font-family: 'DM Sans', sans-serif !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: white !important;
}

.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
}

div[data-testid="metric-container"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

.stAlert { border-radius: 10px !important; }
.stSpinner { color: var(--accent) !important; }

/* Custom cards */
.card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 0.75rem;
}
.card-accent {
    border-left: 3px solid var(--accent);
}
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
</style>
""", unsafe_allow_html=True)

# ─── CLIENTS ──────────────────────────────────────────────────────────────────
@st.cache_resource
def get_clients():
    api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    client = OpenAI(api_key=api_key)
    chroma = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma.get_or_create_collection(
        name="school_booksv2",
        metadata={"hnsw:space": "cosine"}
    )
    return client, collection

SOURCE_MAP = {
    "almaty_7_class": "Информатика, 7 класс — Алматыкітап баспасы, 2021",
}

# ─── SESSION STORAGE (JSON file for persistence across users) ─────────────────
SESSION_FILE = "./sessions.json"

def load_sessions():
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_sessions(data):
    with open(SESSION_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def generate_session_code():
    ts = str(time.time()).encode()
    return hashlib.md5(ts).hexdigest()[:6].upper()

# ─── RAG FUNCTIONS ────────────────────────────────────────────────────────────
def embed(text, client):
    r = client.embeddings.create(model="text-embedding-3-large", input=text)
    v = np.array(r.data[0].embedding)
    return (v / np.linalg.norm(v)).tolist()

def ask(query, collection, client, top_k=5):
    q_emb = embed(query, client)
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    results = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        results.append({"text": doc, "book": meta["book"], "page": meta["page"], "dist": float(dist)})
    return results

def get_context_text(question, collection, client):
    contexts = ask(question, collection, client, top_k=5)
    texts = [c["text"] for c in contexts]
    sources = list({f'{c["book"]}, page {c["page"]}' for c in contexts})
    return "\n\n".join(texts), "; ".join(sources)

def keyword_score(context, answer):
    cw = set(context.lower().split())
    aw = set(answer.lower().split())
    return len(cw & aw) / max(len(cw), 1)

def evaluate_answer(question, student_answer, collection, client):
    context_text, source_text = get_context_text(question, collection, client)
    kw_score = keyword_score(context_text, student_answer)
    ctx = context_text[:3000]
    prompt = f"""Ты система проверки знаний школьников. Отвечай ТОЛЬКО на русском языке.

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
Источник: [info]"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.choices[0].message.content

    # Parse score
    score = 50
    for line in raw.split("\n"):
        if line.strip().startswith("Score:"):
            try:
                score = int(''.join(filter(str.isdigit, line.split(":", 1)[1].strip().split()[0])))
            except:
                pass

    return {"feedback": raw, "score": score, "keyword_score": kw_score, "source": source_text}

def answer_with_llm(query, collection, client, top_k=5):
    contexts = ask(query, collection, client, top_k)
    if not contexts:
        return "Answer not found in the knowledge base."
    sources = {}
    for c in contexts:
        key = (c["book"], c["page"])
        sources[key] = sources.get(key, 0) + 1
    context_text = "\n\n".join([f"[{i+1}] {c['text']}" for i, c in enumerate(contexts)])
    prompt = f"""You are a learning assistant.
Answer ONLY based on the provided textbook fragments.
Cite source numbers in brackets at the end.
If information is insufficient, say so.

Fragments:
{context_text}

Question: {query}

Answer:"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    answer = response.choices[0].message.content.strip()
    src_lines = [f"[{i}] {SOURCE_MAP.get(book, book)}, p.{page}"
                 for i, ((book, page), _) in enumerate(sources.items(), 1)]
    return answer + "\n\nSources:\n" + "\n".join(src_lines)

# ─── XP & LEVELS ──────────────────────────────────────────────────────────────
def score_to_xp(score):
    """Convert 0-100 score to XP. Minimum 5 XP for participation."""
    return max(5, int(score * 1.5))

def xp_to_level(xp):
    thresholds = [0, 50, 150, 300, 500, 750, 1000, 1500, 2000, 3000]
    names = ["Newcomer", "Learner", "Student", "Scholar",
             "Expert", "Master", "Champion", "Legend", "Guru", "Sage"]
    for i in range(len(thresholds) - 1, -1, -1):
        if xp >= thresholds[i]:
            next_xp = thresholds[i+1] if i+1 < len(thresholds) else thresholds[-1]
            return names[i], i+1, thresholds[i], next_xp
    return "Newcomer", 1, 0, 50

def xp_progress_bar(xp):
    _, level, curr_thresh, next_thresh = xp_to_level(xp)
    if next_thresh == curr_thresh:
        pct = 100
    else:
        pct = int((xp - curr_thresh) / (next_thresh - curr_thresh) * 100)
    return pct

# ─── RANK DISPLAY ─────────────────────────────────────────────────────────────
def rank_medal(rank):
    return ["🥇", "🥈", "🥉"][rank - 1] if rank <= 3 else f"#{rank}"

# ─── MAIN APP ─────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem;'>
        <span style='font-family: Space Mono, monospace; font-size: 2rem; font-weight:700;
        background: linear-gradient(135deg, #6366f1, #22d3ee); -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;'>EduQuest</span>
        <span style='color: #64748b; font-size: 0.9rem; margin-left: 0.5rem;'>/ Collaborative Learning Platform</span>
    </div>
    """, unsafe_allow_html=True)

    # Login state
    if "role" not in st.session_state:
        st.session_state.role = None
    if "username" not in st.session_state:
        st.session_state.username = None
    if "session_code" not in st.session_state:
        st.session_state.session_code = None

    # ── LOGIN SCREEN ──────────────────────────────────────────────────────────
    if st.session_state.role is None:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            role = st.radio("I am a...", ["👨‍🏫 Teacher", "👨‍🎓 Student"], horizontal=True)
            st.markdown("<br>", unsafe_allow_html=True)

            if role == "👨‍🏫 Teacher":
                st.markdown("### Teacher Login")
                name = st.text_input("Your name")
                password = st.text_input("Password", type="password", help="Ask admin for password")
                if st.button("Enter as Teacher", use_container_width=True):
                    if password == st.secrets.get("TEACHER_PASSWORD", "teacher123"):
                        st.session_state.role = "teacher"
                        st.session_state.username = name or "Teacher"
                        st.rerun()
                    else:
                        st.error("Wrong password")
            else:
                st.markdown("### Student Login")
                name = st.text_input("Your name")
                code = st.text_input("Session code (from teacher)", placeholder="ABC123")
                if st.button("Join Session", use_container_width=True):
                    if not name:
                        st.error("Enter your name")
                    elif not code:
                        st.error("Enter session code")
                    else:
                        sessions = load_sessions()
                        code_upper = code.strip().upper()
                        if code_upper not in sessions:
                            st.error("Session not found. Check the code.")
                        else:
                            st.session_state.role = "student"
                            st.session_state.username = name
                            st.session_state.session_code = code_upper
                            # Register student in session
                            if name not in sessions[code_upper]["students"]:
                                sessions[code_upper]["students"][name] = {"xp": 0, "answers": []}
                                save_sessions(sessions)
                            st.rerun()
        return

    # ── TEACHER VIEW ──────────────────────────────────────────────────────────
    if st.session_state.role == "teacher":
        teacher_view()

    # ── STUDENT VIEW ──────────────────────────────────────────────────────────
    elif st.session_state.role == "student":
        student_view()

# ─── TEACHER VIEW ─────────────────────────────────────────────────────────────
def teacher_view():
    client, collection = get_clients()
    sessions = load_sessions()

    col_title, col_logout = st.columns([5, 1])
    with col_title:
        st.markdown(f"### 👨‍🏫 Teacher Panel — *{st.session_state.username}*")
    with col_logout:
        if st.button("Logout"):
            st.session_state.role = None
            st.rerun()

    tab1, tab2, tab3 = st.tabs(["🆕 New Session", "📋 Active Sessions", "📊 Analytics"])

    # ── TAB 1: CREATE SESSION ─────────────────────────────────────────────────
    with tab1:
        st.markdown("#### Create a new quiz session")
        col1, col2 = st.columns([3, 2])
        with col1:
            session_title = st.text_input("Session title", placeholder="e.g. Chapter 5 Review")
            question = st.text_area("Question for students", height=120,
                                    placeholder="e.g. What is RAM and how does it work?")
            time_limit = st.slider("Time limit (minutes)", 2, 30, 10)

        with col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.info("💡 Students will answer this question. AI will evaluate their answers using the textbook and assign XP scores.")
            chat_enabled = st.toggle("Allow AI chat during quiz", value=False)
            st.caption("Recommended: OFF during quizzes")

        if st.button("🚀 Create Session & Get Code", use_container_width=True):
            if not question:
                st.error("Write a question first")
            else:
                code = generate_session_code()
                sessions[code] = {
                    "title": session_title or "Quiz Session",
                    "question": question,
                    "time_limit": time_limit,
                    "chat_enabled": chat_enabled,
                    "created_at": datetime.now().isoformat(),
                    "status": "active",
                    "students": {},
                    "teacher": st.session_state.username
                }
                save_sessions(sessions)
                st.success(f"Session created!")
                st.markdown(f"""
                <div style='background: #1a2235; border: 2px solid #6366f1; border-radius: 12px;
                     padding: 1.5rem; text-align: center; margin-top: 1rem;'>
                    <div style='color: #64748b; font-size: 0.9rem; margin-bottom: 0.5rem;'>
                        Share this code with students:
                    </div>
                    <div style='font-family: Space Mono, monospace; font-size: 3rem; font-weight: 700;
                         color: #22d3ee; letter-spacing: 0.3em;'>{code}</div>
                    <div style='color: #64748b; font-size: 0.8rem; margin-top: 0.5rem;'>
                        Session: {session_title or "Quiz Session"}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── TAB 2: ACTIVE SESSIONS ────────────────────────────────────────────────
    with tab2:
        sessions = load_sessions()
        if not sessions:
            st.info("No sessions yet. Create one in the first tab.")
        else:
            for code, sess in sorted(sessions.items(),
                                     key=lambda x: x[1].get("created_at", ""), reverse=True):
                status_color = "#10b981" if sess["status"] == "active" else "#64748b"
                with st.expander(f"**{sess['title']}** — Code: `{code}`  |  {len(sess['students'])} students"):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**Question:** {sess['question']}")
                        st.markdown(f"**Status:** <span style='color:{status_color}'>{sess['status'].upper()}</span>",
                                    unsafe_allow_html=True)
                    with col2:
                        if sess["status"] == "active":
                            if st.button(f"⏹ Close session", key=f"close_{code}"):
                                sessions[code]["status"] = "closed"
                                save_sessions(sessions)
                                st.rerun()
                        else:
                            if st.button(f"🔄 Reopen", key=f"reopen_{code}"):
                                sessions[code]["status"] = "active"
                                save_sessions(sessions)
                                st.rerun()

                    # Leaderboard
                    if sess["students"]:
                        st.markdown("##### 🏆 Leaderboard")
                        sorted_students = sorted(sess["students"].items(),
                                                 key=lambda x: x[1].get("xp", 0), reverse=True)
                        for i, (name, data) in enumerate(sorted_students):
                            xp = data.get("xp", 0)
                            level_name, level_num, _, _ = xp_to_level(xp)
                            answers_count = len(data.get("answers", []))
                            medal = rank_medal(i + 1)
                            st.markdown(f"""
                            <div class='card' style='padding: 0.75rem 1rem;'>
                                <span style='font-size:1.2rem'>{medal}</span>
                                <strong style='margin-left: 0.5rem'>{name}</strong>
                                <span class='xp-badge' style='margin-left: 0.5rem'>⚡ {xp} XP</span>
                                <span class='level-badge' style='margin-left: 0.5rem'>Lv.{level_num} {level_name}</span>
                                <span style='color: #64748b; font-size: 0.8rem; margin-left: 0.5rem'>{answers_count} answers</span>
                            </div>
                            """, unsafe_allow_html=True)

                        # Show all answers
                        st.markdown("##### 📝 Student Answers")
                        for name, data in sess["students"].items():
                            for ans in data.get("answers", []):
                                with st.expander(f"{name} — Score: {ans.get('score', '?')}/100"):
                                    st.write(f"**Answer:** {ans.get('answer', '')}")
                                    st.code(ans.get('feedback', ''), language=None)

    # ── TAB 3: ANALYTICS ──────────────────────────────────────────────────────
    with tab3:
        sessions = load_sessions()
        total_sessions = len(sessions)
        total_students = sum(len(s["students"]) for s in sessions.values())
        all_scores = []
        for s in sessions.values():
            for st_data in s["students"].values():
                for ans in st_data.get("answers", []):
                    if "score" in ans:
                        all_scores.append(ans["score"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Sessions", total_sessions)
        col2.metric("Total Students", total_students)
        col3.metric("Avg Score", f"{sum(all_scores)/len(all_scores):.0f}%" if all_scores else "—")

        if all_scores:
            import pandas as pd
            st.markdown("##### Score Distribution")
            df = pd.DataFrame({"Score": all_scores})
            st.bar_chart(df["Score"].value_counts().sort_index())


# ─── STUDENT VIEW ─────────────────────────────────────────────────────────────
def student_view():
    client, collection = get_clients()
    sessions = load_sessions()
    code = st.session_state.session_code
    name = st.session_state.username

    if code not in sessions:
        st.error("Session not found.")
        st.session_state.role = None
        st.rerun()
        return

    sess = sessions[code]

    # Ensure student exists
    if name not in sess["students"]:
        sess["students"][name] = {"xp": 0, "answers": []}
        save_sessions(sessions)

    student_data = sess["students"][name]
    xp = student_data.get("xp", 0)
    level_name, level_num, curr_thresh, next_thresh = xp_to_level(xp)
    progress_pct = xp_progress_bar(xp)

    # Header bar
    col_title, col_xp, col_logout = st.columns([3, 2, 1])
    with col_title:
        st.markdown(f"### 👋 *{name}*  —  `{code}`")
    with col_xp:
        st.markdown(f"""
        <div style='text-align:right; padding-top: 0.5rem;'>
            <span class='xp-badge'>⚡ {xp} XP</span>
            <span class='level-badge' style='margin-left: 0.5rem'>Lv.{level_num} {level_name}</span>
        </div>
        <div style='background: #1e293b; border-radius: 999px; height: 6px; margin-top: 6px;'>
            <div style='background: linear-gradient(90deg, #6366f1, #22d3ee);
                 width: {progress_pct}%; height: 100%; border-radius: 999px;'></div>
        </div>
        """, unsafe_allow_html=True)
    with col_logout:
        if st.button("Exit"):
            st.session_state.role = None
            st.rerun()

    st.markdown("---")

    # Session status
    if sess["status"] != "active":
        st.warning("⏹ This session has been closed by the teacher.")
    
    tab1, tab2, tab3 = st.tabs(["📝 Answer Question", "🏆 Leaderboard", "💬 AI Chat"])

    # ── TAB 1: ANSWER ─────────────────────────────────────────────────────────
    with tab1:
        st.markdown(f"""
        <div class='card card-accent' style='margin-bottom: 1.5rem;'>
            <div style='color: #64748b; font-size: 0.8rem; text-transform: uppercase;
                 letter-spacing: 0.1em; margin-bottom: 0.5rem;'>📌 Question</div>
            <div style='font-size: 1.1rem; font-weight: 500;'>{sess['question']}</div>
        </div>
        """, unsafe_allow_html=True)

        if sess["status"] != "active":
            st.info("Session is closed. You can still view your results below.")
        else:
            # Проверяем есть ли уже ответ
            past = sessions[code]["students"][name].get("answers", [])

            if past:
                st.success("✅ You have already submitted your answer.")
                # показываем последний результат
                last = past[-1]
                st.metric("Your score", f"{last['score']}/100")
                st.metric("XP earned", f"+{last.get('xp_earned', 0)} XP")
                with st.expander("View feedback"):
                    st.code(last["feedback"], language=None)
            else:
                student_answer = st.text_area(
                "Your answer",
                height=150,
                placeholder="Write your answer here...",
                key="student_answer_input"
            )

            if st.button("✅ Submit Answer", use_container_width=True, disabled=(sess["status"] != "active")):
                if not student_answer.strip():
                    st.warning("Please write your answer first.")
                else:
                    with st.spinner("AI is evaluating your answer..."):
                        result = evaluate_answer(sess["question"], student_answer, collection, client)

                    score = result["score"]
                    xp_earned = score_to_xp(score)

                    # Save to session
                    sessions = load_sessions()
                    sessions[code]["students"][name]["xp"] = (
                        sessions[code]["students"][name].get("xp", 0) + xp_earned
                    )
                    sessions[code]["students"][name]["answers"].append({
                        "answer": student_answer,
                        "score": score,
                        "feedback": result["feedback"],
                        "xp_earned": xp_earned,
                        "submitted_at": datetime.now().isoformat()
                    })
                    save_sessions(sessions)

                    # Score display
                    score_color = "#10b981" if score >= 70 else "#f59e0b" if score >= 40 else "#ef4444"
                    st.markdown(f"""
                    <div style='display: flex; gap: 1rem; margin: 1rem 0;'>
                        <div class='card' style='flex: 1; text-align: center;'>
                            <div style='font-size: 2.5rem; font-weight: 700; color: {score_color};
                                 font-family: Space Mono, monospace;'>{score}</div>
                            <div style='color: #64748b; font-size: 0.85rem;'>/ 100 Score</div>
                        </div>
                        <div class='card' style='flex: 1; text-align: center;'>
                            <div style='font-size: 2.5rem; font-weight: 700; color: #f59e0b;
                                 font-family: Space Mono, monospace;'>+{xp_earned}</div>
                            <div style='color: #64748b; font-size: 0.85rem;'>XP Earned</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander("📋 Detailed Feedback", expanded=True):
                        st.code(result["feedback"], language=None)

                    st.rerun()

        # Past answers
        sessions = load_sessions()
        past = sessions[code]["students"][name].get("answers", [])
        if past:
            st.markdown("##### Your previous answers")
            for i, ans in enumerate(reversed(past), 1):
                sc = ans.get("score", 0)
                xp_e = ans.get("xp_earned", 0)
                sc_color = "#10b981" if sc >= 70 else "#f59e0b" if sc >= 40 else "#ef4444"
                with st.expander(f"Attempt #{len(past)-i+1} — Score: {sc}/100 — +{xp_e} XP"):
                    st.write(f"**Your answer:** {ans.get('answer', '')}")
                    st.code(ans.get("feedback", ""), language=None)

    # ── TAB 2: LEADERBOARD ────────────────────────────────────────────────────
    with tab2:
        sessions = load_sessions()
        sess_fresh = sessions[code]
        st.markdown(f"#### 🏆 {sess_fresh['title']} — Leaderboard")
        st.caption("Refreshes when you switch tabs or resubmit")

        sorted_students = sorted(
            sess_fresh["students"].items(),
            key=lambda x: x[1].get("xp", 0), reverse=True
        )

        if not sorted_students:
            st.info("No students have answered yet.")
        else:
            for i, (sname, sdata) in enumerate(sorted_students):
                sxp = sdata.get("xp", 0)
                slevel_name, slevel_num, _, _ = xp_to_level(sxp)
                medal = rank_medal(i + 1)
                is_me = (sname == name)
                
                if is_me:
                    st.markdown(f"**{medal} {sname}** *(вы)* — ⚡ {sxp} XP — Ур.{slevel_num} {slevel_name}")
                    st.progress(xp_progress_bar(sxp) / 100)
                else:
                    st.markdown(f"**{medal} {sname}** — ⚡ {sxp} XP — Ур.{slevel_num} {slevel_name}")
                    st.progress(xp_progress_bar(sxp) / 100)
                st.divider()

        if st.button("🔄 Refresh Leaderboard"):
            st.rerun()

    # ── TAB 3: AI CHAT ────────────────────────────────────────────────────────
    with tab3:
        sessions = load_sessions()
        chat_allowed = sessions[code].get("chat_enabled", False)

        if not chat_allowed:
            st.markdown("""
            <div style='text-align: center; padding: 3rem; color: #64748b;'>
                <div style='font-size: 3rem;'>🔒</div>
                <div style='font-size: 1.1rem; margin-top: 1rem;'>
                    AI Chat is disabled during this quiz session.
                </div>
                <div style='font-size: 0.85rem; margin-top: 0.5rem;'>
                    Complete your answer first. Chat may be enabled after the quiz.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            if "chat_messages" not in st.session_state:
                st.session_state.chat_messages = []

            for msg in st.session_state.chat_messages:
                st.chat_message(msg["role"]).write(msg["content"])

            user_input = st.chat_input("Ask a question about the textbook...")
            if user_input:
                st.session_state.chat_messages.append({"role": "user", "content": user_input})
                st.chat_message("user").write(user_input)
                with st.spinner("Searching textbook..."):
                    answer = answer_with_llm(user_input, collection, client)
                st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                st.chat_message("assistant").write(answer)


if __name__ == "__main__":
    main()