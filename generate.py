from __future__ import annotations

import argparse
import sys

import psycopg2
import requests

from search import (
    DB_HOST,
    DB_NAME,
    DB_PASSWORD,
    DB_PORT,
    DB_USER,
    OLLAMA_URL,
    EMBEDDING_MODEL,
    get_query_embedding,
    reciprocal_rank_fusion,
    fulltext_search,
    vector_search,
)

GENERATION_MODEL = "gemma4:e4b"

SYSTEM_PROMPT = """Ты - помощник по поиску информации во внутренних документах компании.
Отвечай на русском языке и используй ТОЛЬКО сведения из фрагментов документов,
переданных в контексте. Не добавляй сведения из общих знаний и не делай догадок.

Если в найденных фрагментах нет ответа или данных недостаточно, прямо скажи:
«В найденных документах недостаточно информации, чтобы ответить на вопрос».
Не выдумывай названия документов, номера страниц, сроки, роли или процедуры.
Для каждого существенного утверждения указывай источник в формате
[Источник: название файла, страница/раздел]. Если страница или раздел неизвестны,
укажи только название файла. Не утверждай, что проверил документы, которых нет
в переданном контексте."""


def source_label(item: dict) -> str:
    label = f"{item['title']}.{item['file_type']}"
    if item.get("page_no") is not None:
        label += f", страница {item['page_no']}"
    if item.get("section_title"):
        label += f", раздел «{item['section_title']}»"
    return label


def retrieve_context(question: str, top_k: int) -> list[dict]:
    query_vector = get_query_embedding(question)
    with psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        connect_timeout=10,
    ) as connection:
        vector_results = vector_search(connection, query_vector, top_k)
        text_results = fulltext_search(connection, question, top_k)
        hybrid_results = reciprocal_rank_fusion(vector_results, text_results)
    return hybrid_results[:top_k]


def build_context(results: list[dict]) -> str:
    parts = []
    for number, item in enumerate(results, start=1):
        content = " ".join(str(item["content"]).split())
        parts.append(
            f"ФРАГМЕНТ {number}\n"
            f"Источник: {source_label(item)}\n"
            f"Содержание: {content}"
        )
    return "\n\n---\n\n".join(parts)


def generate_answer(question: str, context: str) -> str:
    response = requests.post(
        f"{OLLAMA_URL.rstrip('/')}/api/chat",
        json={
            "model": GENERATION_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"ВОПРОС СОТРУДНИКА:\n{question}\n\n"
                        f"НАЙДЕННЫЕ ФРАГМЕНТЫ ДОКУМЕНТОВ:\n{context}\n\n"
                        "Сформируй точный ответ по найденным фрагментам. "
                        "Обязательно укажи источники."
                    ),
                },
            ],
            "options": {"temperature": 0.1},
        },
        timeout=(10, 600),
    )
    response.raise_for_status()
    payload = response.json()
    answer = payload.get("message", {}).get("content", "").strip()
    if not answer:
        raise RuntimeError("Ollama вернула пустой ответ.")
    return answer


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ответ на вопрос по документам с использованием гибридного RAG-поиска."
    )
    parser.add_argument("question", nargs="?", help="Вопрос по загруженным документам")
    parser.add_argument(
        "--top-k", type=int, default=4,
        help="Максимальное число фрагментов, передаваемых LLM (по умолчанию: 4)",
    )
    args = parser.parse_args()

    question = (args.question or input("Введите вопрос по документам: ")).strip()
    if not question:
        print("Вопрос не должен быть пустым.", file=sys.stderr)
        return 2
    if args.top_k < 1:
        print("--top-k должен быть не меньше 1.", file=sys.stderr)
        return 2

    try:
        results = retrieve_context(question, args.top_k)
        print(f"Вопрос: {question}")
        print(f"Поиск выполнен: vector + full-text + RRF; embedding: {EMBEDDING_MODEL}")

        if not results:
            print("В найденных документах недостаточно информации, чтобы ответить на вопрос.")
            return 0

        print("\nНайденные источники:")
        for number, item in enumerate(results, start=1):
            print(f"{number}. {source_label(item)}; RRF score: {item['rrf_score']:.6f}")

        context = build_context(results)
        print(f"\nГенеративная модель: {GENERATION_MODEL}")
        print("Формируется ответ... )")
        answer = generate_answer(question, context)
        print("\n" + "-" * 78)
        print("ОТВЕТ ПО ДОКУМЕНТАМ")
        print("-" * 78)
        print(answer)
        return 0

    except requests.RequestException as error:
        print(f"Ошибка обращения к Ollama: {error}", file=sys.stderr)
        return 1
    except (psycopg2.Error, RuntimeError) as error:
        print(f"Ошибка RAG-пайплайна: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
