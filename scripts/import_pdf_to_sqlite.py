# scripts/import_pdf_to_sqlite.py
import sqlite3
import pathlib
import pdfplumber

# ==== 設定部分 ====
# DBは data/courses.db に作る
DB_PATH = pathlib.Path("data") / "courses.db"

# PDFは data/raw 配下の *.pdf を全部読む
PDF_DIR = pathlib.Path("data") / "raw"

# とりあえず 2025年度分として扱う
YEAR = 2025


def create_tables(conn):
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_syllabus (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            year      INTEGER,
            raw_text  TEXT
        );
        """
    )
    conn.commit()


def extract_text_from_pdf(pdf_path: pathlib.Path) -> str:
    """PDFから全ページのテキストを連結して返す"""
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts)


def main():
    # DB接続（ファイルがなければ自動作成）
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)  # data フォルダがなければ作る
    conn = sqlite3.connect(DB_PATH)
    create_tables(conn)
    cur = conn.cursor()

    # PDFフォルダの存在確認（なければ空フォルダ作る）
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    print(f"{len(pdf_files)} 件のPDFを処理する: {PDF_DIR}")

    for pdf_path in pdf_files:
        print(f"処理中: {pdf_path.name}")

        raw_text = extract_text_from_pdf(pdf_path)

        # すでに同じファイル名＋年度が登録されていたらスキップ
        cur.execute(
            "SELECT id FROM raw_syllabus WHERE file_name = ? AND year = ?",
            (pdf_path.name, YEAR),
        )
        row = cur.fetchone()
        if row is not None:
            print(f"  → 既に登録済み (id={row[0]})。スキップ")
            continue

        cur.execute(
            """
            INSERT INTO raw_syllabus (file_name, year, raw_text)
            VALUES (?, ?, ?)
            """,
            (pdf_path.name, YEAR, raw_text),
        )
        conn.commit()
        print("  → 登録完了")

    conn.close()
    print("全PDFの取り込みが完了した")


if __name__ == "__main__":
    main()
