"""Neon PostgreSQL database layer for EduQuest."""
import os
import hashlib
import time
import psycopg2
import psycopg2.extras


def _url() -> str:
    try:
        import streamlit as st
        return st.secrets.get("DATABASE_URL", os.environ.get("DATABASE_URL", ""))
    except Exception:
        return os.environ.get("DATABASE_URL", "")


def _conn():
    return psycopg2.connect(_url(), cursor_factory=psycopg2.extras.RealDictCursor)


def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


# ─── INIT ─────────────────────────────────────────────────────────────────────

def init_db():
    c = _conn()
    try:
        with c.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS teachers (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100) UNIQUE NOT NULL,
                    password_hash CHAR(64) NOT NULL
                );

                CREATE TABLE IF NOT EXISTS students (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100) UNIQUE NOT NULL,
                    grade INTEGER NOT NULL,
                    password_hash CHAR(64) NOT NULL,
                    total_xp INTEGER DEFAULT 0,
                    level INTEGER DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id SERIAL PRIMARY KEY,
                    code CHAR(6) UNIQUE NOT NULL,
                    title VARCHAR(200) NOT NULL,
                    question TEXT,
                    grade INTEGER NOT NULL,
                    lang VARCHAR(2) DEFAULT 'ru',
                    time_limit INTEGER DEFAULT 10,
                    chat_enabled BOOLEAN DEFAULT FALSE,
                    teacher_id INTEGER REFERENCES teachers(id),
                    status VARCHAR(10) DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS questions (
                    id SERIAL PRIMARY KEY,
                    session_id INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
                    question_text TEXT NOT NULL,
                    order_num INTEGER NOT NULL DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS answers (
                    id SERIAL PRIMARY KEY,
                    student_id INTEGER REFERENCES students(id),
                    session_id INTEGER REFERENCES sessions(id),
                    question_id INTEGER REFERENCES questions(id),
                    answer_text TEXT NOT NULL,
                    score INTEGER NOT NULL,
                    xp_earned INTEGER NOT NULL,
                    feedback TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)

            # Add question_id column to answers if missing
            cur.execute("""
                ALTER TABLE answers
                ADD COLUMN IF NOT EXISTS question_id INTEGER REFERENCES questions(id);
            """)

            # Migrate old single-question sessions to questions table
            cur.execute("""
                INSERT INTO questions (session_id, question_text, order_num)
                SELECT s.id, s.question, 1
                FROM sessions s
                WHERE s.question IS NOT NULL AND s.question != ''
                  AND NOT EXISTS (SELECT 1 FROM questions q WHERE q.session_id = s.id)
            """)

        c.commit()
        _seed(c)
    finally:
        c.close()


def _seed(c):
    students = [
        ("Айгерім", 5), ("Берік", 5), ("Нұрлан", 5), ("Зарина", 5),
        ("Арман", 6), ("Малика", 6), ("Дамир", 6), ("Айнур", 6),
        ("Бауыржан", 7), ("Гүлнар", 7), ("Ержан", 7), ("Камила", 7),
        ("Алибек", 8), ("Жанар", 8), ("Мадина", 8), ("Нұрбек", 8),
        ("Асел", 9), ("Болат", 9), ("Динара", 9), ("Ерлан", 9),
        ("Азамат", 10), ("Меруерт", 10), ("Тимур", 10), ("Дариға", 10),
        ("Аян", 11), ("Гульфия", 11), ("Серік", 11), ("Назгүл", 11),
    ]
    with c.cursor() as cur:
        cur.execute(
            "INSERT INTO teachers (name, password_hash) VALUES (%s, %s) ON CONFLICT (name) DO NOTHING",
            ("Askhat", _hash("teacher123"))
        )
        for name, grade in students:
            cur.execute(
                "INSERT INTO students (name, grade, password_hash) VALUES (%s, %s, %s) ON CONFLICT (name) DO NOTHING",
                (name, grade, _hash("student123"))
            )
    c.commit()


# ─── AUTH ─────────────────────────────────────────────────────────────────────

def verify_teacher(name: str, password: str):
    c = _conn()
    try:
        with c.cursor() as cur:
            cur.execute(
                "SELECT id, name FROM teachers WHERE name=%s AND password_hash=%s",
                (name, _hash(password))
            )
            return cur.fetchone()
    finally:
        c.close()


def verify_student(name: str, password: str):
    c = _conn()
    try:
        with c.cursor() as cur:
            cur.execute(
                "SELECT id, name, grade, total_xp, level FROM students WHERE name=%s AND password_hash=%s",
                (name, _hash(password))
            )
            return cur.fetchone()
    finally:
        c.close()


def get_student_info(student_id: int):
    c = _conn()
    try:
        with c.cursor() as cur:
            cur.execute(
                "SELECT id, name, grade, total_xp, level FROM students WHERE id=%s",
                (student_id,)
            )
            return cur.fetchone()
    finally:
        c.close()


# ─── SESSIONS ─────────────────────────────────────────────────────────────────

def _expire():
    c = _conn()
    try:
        with c.cursor() as cur:
            cur.execute("""
                UPDATE sessions SET status = 'closed'
                WHERE status = 'active'
                  AND created_at + (time_limit || ' minutes')::interval < NOW()
            """)
        c.commit()
    finally:
        c.close()


def create_session(title, question_texts: list, grade, lang, time_limit, teacher_id):
    """Create session with a list of questions."""
    code = hashlib.md5(str(time.time()).encode()).hexdigest()[:6].upper()
    first_q = question_texts[0] if question_texts else ""
    c = _conn()
    try:
        with c.cursor() as cur:
            cur.execute("""
                INSERT INTO sessions (code, title, question, grade, lang, time_limit, chat_enabled, teacher_id)
                VALUES (%s, %s, %s, %s, %s, %s, FALSE, %s) RETURNING id, code
            """, (code, title, first_q, grade, lang, time_limit, teacher_id))
            row = cur.fetchone()
            session_id = row["id"]

            for i, text in enumerate(question_texts, 1):
                cur.execute(
                    "INSERT INTO questions (session_id, question_text, order_num) VALUES (%s, %s, %s)",
                    (session_id, text.strip(), i)
                )
        c.commit()
        return row
    finally:
        c.close()


def get_session_questions(session_id: int):
    c = _conn()
    try:
        with c.cursor() as cur:
            cur.execute(
                "SELECT * FROM questions WHERE session_id=%s ORDER BY order_num",
                (session_id,)
            )
            return cur.fetchall()
    finally:
        c.close()


def get_sessions_for_grade(grade: int):
    _expire()
    c = _conn()
    try:
        with c.cursor() as cur:
            cur.execute("""
                SELECT s.*, t.name as teacher_name,
                       GREATEST(0, EXTRACT(EPOCH FROM
                           (s.created_at + (s.time_limit || ' minutes')::interval - NOW())
                       ) / 60)::int AS mins_left,
                       (SELECT COUNT(*) FROM questions q WHERE q.session_id = s.id) AS question_count
                FROM sessions s
                JOIN teachers t ON t.id = s.teacher_id
                WHERE s.grade = %s AND s.status = 'active'
                ORDER BY s.created_at DESC
            """, (grade,))
            return cur.fetchall()
    finally:
        c.close()


def get_session(session_id: int):
    _expire()
    c = _conn()
    try:
        with c.cursor() as cur:
            cur.execute(
                "SELECT s.*, (SELECT COUNT(*) FROM questions q WHERE q.session_id = s.id) AS question_count FROM sessions s WHERE s.id=%s",
                (session_id,)
            )
            return cur.fetchone()
    finally:
        c.close()


def get_all_sessions():
    _expire()
    c = _conn()
    try:
        with c.cursor() as cur:
            cur.execute("""
                SELECT s.*, t.name as teacher_name,
                       COUNT(DISTINCT a.student_id) as student_count,
                       (SELECT COUNT(*) FROM questions q WHERE q.session_id = s.id) AS question_count
                FROM sessions s
                JOIN teachers t ON t.id = s.teacher_id
                LEFT JOIN answers a ON a.session_id = s.id
                GROUP BY s.id, t.name
                ORDER BY s.created_at DESC
            """)
            return cur.fetchall()
    finally:
        c.close()


def set_session_status(session_id: int, status: str):
    c = _conn()
    try:
        with c.cursor() as cur:
            if status == 'active':
                cur.execute(
                    "UPDATE sessions SET status=%s, created_at=NOW() WHERE id=%s",
                    (status, session_id)
                )
            else:
                cur.execute("UPDATE sessions SET status=%s WHERE id=%s", (status, session_id))
        c.commit()
    finally:
        c.close()


# ─── ANSWERS ──────────────────────────────────────────────────────────────────

def get_student_answer_for_question(student_id: int, question_id: int):
    c = _conn()
    try:
        with c.cursor() as cur:
            cur.execute("""
                SELECT * FROM answers
                WHERE student_id=%s AND question_id=%s
                LIMIT 1
            """, (student_id, question_id))
            return cur.fetchone()
    finally:
        c.close()


def get_student_session_answers(student_id: int, session_id: int):
    """All answers by a student in a session."""
    c = _conn()
    try:
        with c.cursor() as cur:
            cur.execute("""
                SELECT a.*, q.question_text, q.order_num
                FROM answers a
                JOIN questions q ON q.id = a.question_id
                WHERE a.student_id=%s AND a.session_id=%s
                ORDER BY q.order_num
            """, (student_id, session_id))
            return cur.fetchall()
    finally:
        c.close()


def submit_answer(student_id: int, session_id: int, question_id: int,
                  answer_text: str, score: int, feedback: str):
    """Store answer. XP is awarded separately via finalize_session_xp."""
    c = _conn()
    try:
        with c.cursor() as cur:
            cur.execute("""
                INSERT INTO answers (student_id, session_id, question_id, answer_text, score, xp_earned, feedback)
                VALUES (%s, %s, %s, %s, %s, 0, %s)
            """, (student_id, session_id, question_id, answer_text, score, feedback))
        c.commit()
    finally:
        c.close()


def finalize_session_xp(student_id: int, session_id: int, xp: int) -> int:
    """Award XP once at end of session based on avg score. Updates last answer row."""
    c = _conn()
    try:
        with c.cursor() as cur:
            cur.execute("""
                UPDATE answers SET xp_earned = %s
                WHERE id = (
                    SELECT id FROM answers
                    WHERE student_id=%s AND session_id=%s
                    ORDER BY created_at DESC LIMIT 1
                )
            """, (xp, student_id, session_id))
            cur.execute(
                "UPDATE students SET total_xp = total_xp + %s WHERE id=%s RETURNING total_xp",
                (xp, student_id)
            )
            new_xp = cur.fetchone()["total_xp"]
            cur.execute("UPDATE students SET level=%s WHERE id=%s", (_calc_level(new_xp), student_id))
        c.commit()
        return new_xp
    finally:
        c.close()


def _calc_level(xp: int) -> int:
    thresholds = [0, 50, 150, 300, 500, 750, 1000, 1500, 2000, 3000]
    for i in range(len(thresholds) - 1, -1, -1):
        if xp >= thresholds[i]:
            return i + 1
    return 1


# ─── LEADERBOARDS ─────────────────────────────────────────────────────────────

def get_session_leaderboard(session_id: int):
    c = _conn()
    try:
        with c.cursor() as cur:
            cur.execute("""
                SELECT s.name, s.grade,
                       SUM(a.xp_earned) as session_xp,
                       ROUND(AVG(a.score)) as avg_score,
                       COUNT(a.id) as answered
                FROM answers a
                JOIN students s ON s.id = a.student_id
                WHERE a.session_id = %s
                GROUP BY s.id, s.name, s.grade
                ORDER BY session_xp DESC
            """, (session_id,))
            return cur.fetchall()
    finally:
        c.close()


def get_class_leaderboard(grade: int):
    c = _conn()
    try:
        with c.cursor() as cur:
            cur.execute("""
                SELECT name, total_xp, level
                FROM students WHERE grade=%s
                ORDER BY total_xp DESC
            """, (grade,))
            return cur.fetchall()
    finally:
        c.close()


# ─── EXPORT ───────────────────────────────────────────────────────────────────

def get_session_export(session_id: int):
    """Returns detailed rows: one per student per question."""
    c = _conn()
    try:
        with c.cursor() as cur:
            cur.execute("""
                SELECT s.name      AS "Ученик",
                       s.grade     AS "Класс",
                       q.order_num AS "№",
                       q.question_text AS "Вопрос",
                       a.answer_text   AS "Ответ",
                       a.score         AS "Оценка",
                       a.xp_earned     AS "XP",
                       a.feedback      AS "Обратная связь",
                       a.created_at    AS "Время"
                FROM answers a
                JOIN students s ON s.id = a.student_id
                JOIN questions q ON q.id = a.question_id
                WHERE a.session_id = %s
                ORDER BY s.name, q.order_num
            """, (session_id,))
            return cur.fetchall()
    finally:
        c.close()


def get_session_summary_export(session_id: int):
    """Summary: one row per student with totals."""
    c = _conn()
    try:
        with c.cursor() as cur:
            cur.execute("""
                SELECT s.name            AS "Ученик",
                       s.grade           AS "Класс",
                       COUNT(a.id)       AS "Отвечено вопросов",
                       ROUND(AVG(a.score)) AS "Средняя оценка",
                       SUM(a.xp_earned)  AS "Итого XP"
                FROM answers a
                JOIN students s ON s.id = a.student_id
                WHERE a.session_id = %s
                GROUP BY s.id, s.name, s.grade
                ORDER BY SUM(a.xp_earned) DESC
            """, (session_id,))
            return cur.fetchall()
    finally:
        c.close()
