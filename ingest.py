"""Ingest PDF/DOCX files into PostgreSQL + pgvector using Ollama embeddings."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import psycopg2
import requests
from docx import Document as DocxDocument
from docx.document import Document as DocxDocumentObject
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader


BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
DB_HOST = os.getenv("RAG_DB_HOST", "localhost")
DB_PORT = int(os.getenv("RAG_DB_PORT", "5433"))
DB_NAME = os.getenv("RAG_DB_NAME", "rag")
DB_USER = os.getenv("RAG_DB_USER", "rag_user")
DB_PASSWORD = os.getenv("RAG_DB_PASSWORD", "rag_dev_only")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_DIMENSIONS = 768
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180
EMBEDDING_BATCH_SIZE = 24


@dataclass
class TextUnit:
    text: str
    page_no: int | None = None
    section_title: str | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_pdf(path: Path) -> list[TextUnit]:
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        raise ValueError("PDF защищён паролем; зашифрованные документы этим скриптом не обрабатываются.")

    units: list[TextUnit] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            units.append(TextUnit(text=text, page_no=page_number))
    return units


def iter_docx_blocks(document: DocxDocumentObject) -> Iterator[Paragraph | Table]:
    """Yield paragraphs and tables in their original document order."""
    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


def read_docx(path: Path) -> list[TextUnit]:
    document = DocxDocument(str(path))
    units: list[TextUnit] = []
    current_section: str | None = None

    for block in iter_docx_blocks(document):
        if isinstance(block, Paragraph):
            text = clean_text(block.text)
            if not text:
                continue
            style_name = (block.style.name or "") if block.style else ""
            if style_name.startswith("Heading") or style_name == "Title":
                current_section = text
                continue
            units.append(TextUnit(text=text, section_title=current_section))
        else:
            rows: list[str] = []
            for row in block.rows:
                cells = [clean_text(cell.text) for cell in row.cells]
                rows.append(" | ".join(value for value in cells if value))
            table_text = clean_text("\n".join(row for row in rows if row))
            if table_text:
                units.append(TextUnit(text=table_text, section_title=current_section))

    return units


def split_long_text(text: str, max_chars: int) -> list[str]:
    """Split long paragraphs near whitespace boundaries."""
    remaining = text.strip()
    parts: list[str] = []
    while len(remaining) > max_chars:
        cut = remaining.rfind(" ", 0, max_chars + 1)
        if cut < max_chars // 2:
            cut = max_chars
        parts.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def make_chunks(units: list[TextUnit]) -> list[TextUnit]:
    chunks: list[TextUnit] = []
    current_text = ""
    current_page: int | None = None
    current_section: str | None = None

    def flush() -> None:
        nonlocal current_text, current_page, current_section
        value = clean_text(current_text)
        if value:
            chunks.append(TextUnit(value, current_page, current_section))
        current_text = ""

    for unit in units:
        text = clean_text(unit.text)
        if not text:
            continue
        for part in split_long_text(text, CHUNK_SIZE):
            if current_text and len(current_text) + len(part) + 1 > CHUNK_SIZE:
                previous = current_text
                flush()
                overlap = previous[-CHUNK_OVERLAP:].strip()
                current_text = f"{overlap}\n{part}" if overlap else part
                current_page = unit.page_no
                current_section = unit.section_title or current_section
            else:
                current_text = f"{current_text}\n{part}" if current_text else part
                if current_page is None:
                    current_page = unit.page_no
                if unit.section_title:
                    current_section = unit.section_title

    flush()
    return chunks


def extract_units(path: Path) -> list[TextUnit]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix == ".docx":
        return read_docx(path)
    raise ValueError(f"Неподдерживаемый формат: {suffix}")


def get_embeddings(texts: list[str]) -> list[list[float]]:
    all_embeddings: list[list[float]] = []
    endpoint = f"{OLLAMA_URL.rstrip('/')}/api/embed"

    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        response = requests.post(
            endpoint,
            json={"model": EMBEDDING_MODEL, "input": batch},
            timeout=300,
        )
        response.raise_for_status()
        payload = response.json()
        embeddings = payload.get("embeddings", [])
        if len(embeddings) != len(batch):
            raise RuntimeError("Ollama вернула неожиданное количество embeddings.")
        for vector in embeddings:
            if len(vector) != EMBEDDING_DIMENSIONS:
                raise RuntimeError(
                    f"Ожидалось {EMBEDDING_DIMENSIONS} измерений, получено {len(vector)}."
                )
            all_embeddings.append(vector)

    return all_embeddings


def connect_db():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        connect_timeout=10,
    )


def ingest_file(path: Path, connection) -> tuple[str, int]:
    resolved_path = path.resolve()
    source_uri = resolved_path.as_uri()
    checksum = sha256_file(path)
    modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    file_type = path.suffix.lower().lstrip(".")
    title = path.stem

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT doc_id, checksum_sha256 FROM documents WHERE source_uri = %s",
            (source_uri,),
        )
        existing = cursor.fetchone()

        if existing and existing[1] == checksum:
            return "пропущен (файл не изменился)", 0

        units = extract_units(path)
        chunks = make_chunks(units)
        if not chunks:
            raise ValueError("Не удалось извлечь текст. Возможно, PDF является сканом без текстового слоя.")

        embeddings = get_embeddings([chunk.text for chunk in chunks])

        if existing:
            doc_id = existing[0]
            cursor.execute(
                """UPDATE documents
                   SET title = %s, file_type = %s, checksum_sha256 = %s,
                       modified_at = %s, ingested_at = NOW(), is_active = TRUE
                   WHERE doc_id = %s""",
                (title, file_type, checksum, modified_at, doc_id),
            )
            cursor.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
        else:
            cursor.execute(
                """INSERT INTO documents
                   (source_uri, title, file_type, checksum_sha256, modified_at)
                   VALUES (%s, %s, %s, %s, %s)
                   RETURNING doc_id""",
                (source_uri, title, file_type, checksum, modified_at),
            )
            doc_id = cursor.fetchone()[0]

        for chunk_number, (chunk, vector) in enumerate(zip(chunks, embeddings), start=1):
            cursor.execute(
                """INSERT INTO chunks
                   (doc_id, chunk_no, page_no, section_title, content, embedding, embedding_model)
                   VALUES (%s, %s, %s, %s, %s, %s::vector, %s)""",
                (
                    doc_id,
                    chunk_number,
                    chunk.page_no,
                    chunk.section_title,
                    chunk.text,
                    json.dumps(vector),
                    EMBEDDING_MODEL,
                ),
            )

    return "загружен", len(chunks)


def main() -> int:
    DOCUMENTS_DIR.mkdir(exist_ok=True)
    files = sorted(
        path for path in DOCUMENTS_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pdf", ".docx"}
    )

    if not files:
        print(f"В папке {DOCUMENTS_DIR} не найдены PDF или DOCX файлы")
        return 0

    try:
        with connect_db() as connection:
            for path in files:
                try:
                    status, chunk_count = ingest_file(path, connection)
                    connection.commit()
                    print(f"{path.name}: {status}; фрагментов: {chunk_count}")
                except Exception as error:
                    connection.rollback()
                    print(f"ОШИБКА для {path.name}: {error}", file=sys.stderr)
        print("Обработка документов завершена.")
        return 0
    except psycopg2.Error as error:
        print(f"Не удалось подключиться к PostgreSQL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
