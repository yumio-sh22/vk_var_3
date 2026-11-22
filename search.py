"""Vector, full-text and hybrid (RRF) search over ingested PDF/DOCX chunks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import psycopg2
import requests

DB_HOST = os.getenv("RAG_DB_HOST", "localhost")
DB_PORT = int(os.getenv("RAG_DB_PORT", "5433"))
DB_NAME = os.getenv("RAG_DB_NAME", "rag")
DB_USER = os.getenv("RAG_DB_USER", "rag_user")
DB_PASSWORD = os.getenv("RAG_DB_PASSWORD", "rag_dev_only")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_DIMENSIONS = 768
RRF_K = 60

BASE_SELECT = """
    SELECT
        c.chunk_id,
        d.title,
        d.file_type,
        c.page_no,
        c.section_title,
        c.content
    FROM chunks AS c
    JOIN documents AS d ON d.doc_id = c.doc_id
    WHERE d.is_active = TRUE
"""


def get_query_embedding(query: str) -> list[float]:
    response = requests.post(
        f"{OLLAMA_URL.rstrip('/')}/api/embed",
        json={"model": EMBEDDING_MODEL, "input": query},
        timeout=300,
    )
    response.raise_for_status()
    payload = response.json()
    embeddings = payload.get("embeddings", [])
    if not embeddings:
        raise RuntimeError("Ollama не вернула embedding для вопроса.")
    vector = embeddings[0]
    if len(vector) != EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            f"Ожидалось {EMBEDDING_DIMENSIONS} измерений, получено {len(vector)}."
        )
    return vector


def vector_search(connection, vector: list[float], limit: int) -> list[dict[str, Any]]:
    vector_json = json.dumps(vector)
    sql = """
        SELECT
            c.chunk_id,
            d.title,
            d.file_type,
            c.page_no,
            c.section_title,
            c.content,
            1 - (c.embedding <=> %s::vector) AS similarity
        FROM chunks AS c
        JOIN documents AS d ON d.doc_id = c.doc_id
        WHERE d.is_active = TRUE AND c.embedding IS NOT NULL
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
    """
    with connection.cursor() as cursor:
        cursor.execute(sql, (vector_json, vector_json, limit))
        rows = cursor.fetchall()
        columns = [item[0] for item in cursor.description]
    return [dict(zip(columns, row)) for row in rows]


def fulltext_search(connection, query: str, limit: int) -> list[dict[str, Any]]:
    sql = BASE_SELECT + """
        AND c.search_vector @@ websearch_to_tsquery('simple', %s)
        ORDER BY ts_rank_cd(
            c.search_vector,
            websearch_to_tsquery('simple', %s)
        ) DESC
        LIMIT %s
    """
    with connection.cursor() as cursor:
        cursor.execute(sql, (query, query, limit))
        rows = cursor.fetchall()
        columns = [item[0] for item in cursor.description]
    return [dict(zip(columns, row)) for row in rows]


def reciprocal_rank_fusion(
    vector_results: list[dict[str, Any]],
    text_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fused: dict[int, dict[str, Any]] = {}
    for results in (vector_results, text_results):
        for rank, result in enumerate(results, start=1):
            chunk_id = int(result["chunk_id"])
            if chunk_id not in fused:
                fused[chunk_id] = dict(result)
                fused[chunk_id]["rrf_score"] = 0.0
                fused[chunk_id]["vector_rank"] = None
                fused[chunk_id]["text_rank"] = None
            fused[chunk_id]["rrf_score"] += 1.0 / (RRF_K + rank)
            if results is vector_results:
                fused[chunk_id]["vector_rank"] = rank
                fused[chunk_id]["similarity"] = result.get("similarity")
            else:
                fused[chunk_id]["text_rank"] = rank
    return sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)


def print_results(title: str, results: list[dict[str, Any]], include_similarity: bool = False) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    if not results:
        print("Подходящие фрагменты не найдены.")
        return

    for position, item in enumerate(results, start=1):
        source = f"{item['title']}.{item['file_type']}"
        details = []
        if item.get("page_no") is not None:
            details.append(f"страница {item['page_no']}")
        if item.get("section_title"):
            details.append(f"раздел: {item['section_title']}")
        source_line = source + ("; " + "; ".join(details) if details else "")

        print(f"\n{position}. {source_line}")
        if include_similarity and item.get("similarity") is not None:
            print(f"   cosine similarity: {item['similarity']:.4f}")
        if item.get("rrf_score") is not None:
            print(
                "   RRF score: "
                f"{item['rrf_score']:.6f}; "
                f"vector rank: {item.get('vector_rank')}; "
                f"full-text rank: {item.get('text_rank')}"
            )
        text = " ".join(str(item["content"]).split())
        print(f"   {text[:700]}{'...' if len(text) > 700 else ''}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Сравнение векторного, полнотекстового и гибридного поиска по документам."
    )
    parser.add_argument("query", nargs="?", help="Вопрос или поисковая фраза")
    parser.add_argument("--top-k", type=int, default=5, help="Число результатов каждого поиска")
    args = parser.parse_args()

    query = (args.query or input("Введите вопрос по документам: ")).strip()
    if not query:
        print("Вопрос не должен быть пустым.", file=sys.stderr)
        return 2
    if args.top_k < 1:
        print("--top-k должен быть не меньше 1.", file=sys.stderr)
        return 2

    try:
        vector = get_query_embedding(query)
        with psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            connect_timeout=10,
        ) as connection:
            vectors = vector_search(connection, vector, args.top_k)
            lexical = fulltext_search(connection, query, args.top_k)
            hybrid = reciprocal_rank_fusion(vectors, lexical)

        print(f"Вопрос: {query}")
        print(f"Embedding-модель: {EMBEDDING_MODEL}; размерность: {len(vector)}")
        print_results("1.ВЕКТОРНЫЙ ПОИСК", vectors, True)
        print_results("2.ПОЛНОТЕКСТОВЫЙ ПОИСК", lexical)
        print_results("3.ГИБРИДНЫЙ ПОИСК", hybrid)
        return 0
    except (requests.RequestException, psycopg2.Error, RuntimeError) as error:
        print(f"Ошибка поиска: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
