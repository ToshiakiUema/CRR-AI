# scripts/parse_raw_to_courses.py
#
# pdfplumberのextract_table()で1ページ目の表を読む版
# 追加: raw_textから達成目標(learning_goals)と授業計画(course_plan)も抽出してcoursesへ保存

import sqlite3
from pathlib import Path
import re
import pdfplumber

DB_PATH = Path("data") / "courses.db"
PDF_DIR = Path("data") / "raw"

_COMPAT_MAP = str.maketrans({
    "⼈": "人", "⼊": "入", "⼒": "力", "⼼": "心",
    "⽂": "文", "⽅": "方", "⽇": "日", "⽕": "火",
    "⽣": "生", "⽬": "目",
    "⾃": "自", "⾏": "行", "⾒": "見",
    "⾕": "谷", "⾨": "門",
})

def norm_ws(s: str) -> str:
    s = s.translate(_COMPAT_MAP)
    s = s.replace("　", " ")
    return " ".join(s.split()).strip()

def create_courses_table(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS courses (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_id          INTEGER,
            code            TEXT,
            subject_number  TEXT,
            title           TEXT,
            teacher         TEXT,
            year            INTEGER,
            term_label      TEXT,
            semester        TEXT,
            day_period      TEXT,
            faculty         TEXT,
            classroom       TEXT,
            capacity        INTEGER,
            reg_method      TEXT,
            credits         REAL,
            description     TEXT,
            learning_goals  TEXT,
            course_plan     TEXT,
            embedding       TEXT
        );
        """
    )
    conn.commit()

def extract_table_metadata_from_pdf(pdf_path: Path) -> dict:
    meta: dict[str, str] = {}

    if not pdf_path.exists():
        print(f"[WARN] PDF not found: {pdf_path}")
        return meta

    with pdfplumber.open(pdf_path) as pdf:
        table = pdf.pages[0].extract_table()

    rows = table or []

    for i in range(0, len(rows) - 1, 2):
        header_row = rows[i]
        value_row = rows[i + 1]
        if not header_row or not value_row:
            continue

        for col, h in enumerate(header_row):
            if not h:
                continue
            v = value_row[col] if col < len(value_row) else None
            if v is None or v == "":
                continue
            kh = norm_ws(h)
            kv = norm_ws(v)
            meta[kh] = kv

    for k in list(meta.keys()):
        if k.startswith("担当教員"):
            t = re.sub(r"\[.*?\]", "", meta[k])
            meta["担当教員"] = norm_ws(t)

    return meta

def _split_lines(raw_text: str) -> list[str]:
    lines_raw = raw_text.splitlines()
    return [norm_ws(ln) for ln in lines_raw]

def extract_section_text(raw_text: str, start_heading: str, end_headings: list[str]) -> str | None:
    """
    raw_textを行単位で見て、start_headingを含む行の次から、
    end_headingsのどれかが出るまでを抽出する。
    """
    lines_raw = raw_text.splitlines()
    lines_norm = [norm_ws(ln) for ln in lines_raw]

    start_idx = None
    for i, ln in enumerate(lines_norm):
        if start_heading in ln:
            start_idx = i + 1
            break

    if start_idx is None:
        return None

    collected: list[str] = []
    for raw_ln, norm_ln in zip(lines_raw[start_idx:], lines_norm[start_idx:]):
        if any(h in norm_ln for h in end_headings):
            break
        if norm_ln == "":
            collected.append(raw_ln)
            continue
        collected.append(raw_ln)

    text = "\n".join(collected).strip()
    return text or None

def extract_description_from_text(raw_text: str) -> str | None:
    end_keywords = [
        "URGCC学習教育目標",
        "達成目標",
        "評価基準と評価方法",
        "履修条件",
        "授業計画",
        "事前学習",
        "事後学習",
    ]
    return extract_section_text(raw_text, "授業内容と方法", end_keywords) or raw_text.strip() or None

def extract_learning_goals_from_text(raw_text: str) -> str | None:
    end_keywords = [
        "評価基準と評価方法",
        "履修条件",
        "授業計画",
        "事前学習",
        "事後学習",
    ]
    return extract_section_text(raw_text, "達成目標", end_keywords)

def extract_course_plan_from_text(raw_text: str) -> str | None:
    end_keywords = [
        "事前学習",
        "事後学習",
        "教科書",
        "参考文献",
        "メッセージ",
        "教材にかかわる情報",
        "評価基準と評価方法",
        "履修条件",
    ]
    return extract_section_text(raw_text, "授業計画", end_keywords)

def normalize_semester(term_label: str | None) -> str | None:
    if not term_label:
        return None
    if any(k in term_label for k in ["春学期", "前期"]):
        return "spring"
    if any(k in term_label for k in ["秋学期", "後学期", "後期"]):
        return "fall"
    if "通年" in term_label:
        return "full"
    return None

def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    create_courses_table(conn)
    cur = conn.cursor()

    cur.execute("SELECT id, year, file_name, raw_text FROM raw_syllabus")
    rows = cur.fetchall()

    inserted = 0

    for raw_id, year_from_raw, file_name, raw_text in rows:
        pdf_path = PDF_DIR / file_name

        meta = extract_table_metadata_from_pdf(pdf_path)

        description = extract_description_from_text(raw_text)
        learning_goals = extract_learning_goals_from_text(raw_text)
        course_plan = extract_course_plan_from_text(raw_text)

        subject_number = meta.get("科目番号")
        classroom = meta.get("教室")
        capacity_str = meta.get("登録人数")
        reg_method = meta.get("履修登録方法")
        year_str = meta.get("開講年度")
        term_label = meta.get("期間")
        day_period = meta.get("曜日時限")
        faculty = meta.get("開講学部等")
        code = meta.get("講義コード")
        title = meta.get("科目名") or meta.get("科目名[英文名]")
        teacher = meta.get("担当教員")
        credits_str = meta.get("単位数")

        year = year_from_raw
        if year_str and str(year_str).isdigit():
            year = int(year_str)

        semester = normalize_semester(term_label)

        capacity = None
        if capacity_str and str(capacity_str).isdigit():
            capacity = int(capacity_str)

        credits = None
        if credits_str:
            cs = str(credits_str).replace("単位", "")
            try:
                credits = float(cs)
            except ValueError:
                credits = None

        cur.execute("SELECT id FROM courses WHERE raw_id = ?", (raw_id,))
        if cur.fetchone() is not None:
            continue

        cur.execute(
            """
            INSERT INTO courses (
                raw_id, code, subject_number, title, teacher,
                year, term_label, semester, day_period, faculty,
                classroom, capacity, reg_method, credits,
                description, learning_goals, course_plan
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                raw_id,
                code,
                subject_number,
                title,
                teacher,
                year,
                term_label,
                semester,
                day_period,
                faculty,
                classroom,
                capacity,
                reg_method,
                credits,
                description,
                learning_goals,
                course_plan,
            ),
        )
        inserted += 1

    conn.commit()
    conn.close()
    print(f"{inserted}件のコースを追加した")

if __name__ == "__main__":
    main()
