"""Evaluate vector, full-text and hybrid retrieval on a small labelled test set."""

from __future__ import annotations

import math
from collections import defaultdict

import psycopg2

from search import (
    DB_HOST,
    DB_NAME,
    DB_PASSWORD,
    DB_PORT,
    DB_USER,
    get_query_embedding,
    reciprocal_rank_fusion,
    fulltext_search,
    vector_search,
)

TEST_CASES = [
    {
        "question": "Владелец документа проверяет обоснование",
        "relevant_title": "regulation",
    },
    {
        "question": "Что должен сделать владелец после получения запроса на доступ к документу?",
        "relevant_title": "regulation",
    },
    {
        "question": "Сотрудник регистрирует заявку через внутренний портал",
        "relevant_title": "instruction",
    },
    {
        "question": "Какой номер нужно сохранить после отправки заявки?",
        "relevant_title": "instruction",
    },
    {
        "question": "В запросе указываются название документа, рабочая задача и необходимый срок доступа",
        "relevant_title": "regulation",
    },
    {
        "question": "Ответственный специалист проверяет заявку в течение двух рабочих дней",
        "relevant_title": "instruction",
    },
]

NO_ANSWER_QUESTION = "Как оформить командировку в Японию и какой лимит расходов установлен?"
TOP_K = 2


def ranked_titles(results: list[dict]) -> list[str]:
    return [str(item["title"]) for item in results]


def metrics(results: list[dict], relevant_title: str, k: int = TOP_K) -> dict[str, float]:
    titles = ranked_titles(results)
    top_k = titles[:k]
    relevant_count = sum(title == relevant_title for title in top_k)
    precision_at_1 = 1.0 if titles and titles[0] == relevant_title else 0.0
    precision_at_k = relevant_count / k
    recall_at_k = 1.0 if relevant_title in top_k else 0.0

    reciprocal_rank = 0.0
    dcg = 0.0
    for rank, title in enumerate(top_k, start=1):
        if title == relevant_title:
            reciprocal_rank = 1.0 / rank
            dcg = 1.0 / math.log2(rank + 1)
            break
    ndcg_at_k = dcg

    return {
        f"P@1": precision_at_1,
        f"P@{k}": precision_at_k,
        f"Recall@{k}": recall_at_k,
        "MRR": reciprocal_rank,
        f"nDCG@{k}": ndcg_at_k,
    }


def main() -> int:
    method_names = ("vector", "full-text", "hybrid")
    totals: dict[str, dict[str, float]] = {
        method: defaultdict(float) for method in method_names
    }

    print("Оценка поиска на размеченном контрольном наборе")
    print(f"Число вопросов с известным релевантным документом: {len(TEST_CASES)}")
    print(f"k = {TOP_K}\n")

    try:
        with psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            connect_timeout=10,
        ) as connection:
            for index, case in enumerate(TEST_CASES, start=1):
                question = case["question"]
                relevant = case["relevant_title"]
                query_vector = get_query_embedding(question)

                vector_results = vector_search(connection, query_vector, TOP_K)
                text_results = fulltext_search(connection, question, TOP_K)
                hybrid_results = reciprocal_rank_fusion(vector_results, text_results)[:TOP_K]

                result_sets = {
                    "vector": vector_results,
                    "full-text": text_results,
                    "hybrid": hybrid_results,
                }

                print(f"{index}. Вопрос: {question}")
                print(f"   Эталонный документ: {relevant}")
                for method, results in result_sets.items():
                    values = metrics(results, relevant)
                    for metric_name, metric_value in values.items():
                        totals[method][metric_name] += metric_value
                    first = results[0]["title"] if results else "нет результатов"
                    print(
                        f"   {method:9s}: top-1={first}; "
                        f"P@1={values['P@1']:.3f}; "
                        f"P@{TOP_K}={values[f'P@{TOP_K}']:.3f}; "
                        f"Recall@{TOP_K}={values[f'Recall@{TOP_K}']:.3f}; "
                        f"MRR={values['MRR']:.3f}; "
                        f"nDCG@{TOP_K}={values[f'nDCG@{TOP_K}']:.3f}"
                    )
                print()

        count = len(TEST_CASES)
        print("СРЕДНИЕ МАКРО-МЕТРИКИ")
        print("Метод      P@1    P@2    Recall@2   MRR    nDCG@2")
        for method in method_names:
            avg = {name: value / count for name, value in totals[method].items()}
            print(
                f"{method:9s} "
                f"{avg['P@1']:.3f}  "
                f"{avg[f'P@{TOP_K}']:.3f}  "
                f"{avg[f'Recall@{TOP_K}']:.3f}      "
                f"{avg['MRR']:.3f}  "
                f"{avg[f'nDCG@{TOP_K}']:.3f}"
            )

        no_answer_vector = get_query_embedding(NO_ANSWER_QUESTION)
        with psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            connect_timeout=10,
        ) as connection:
            no_answer_vectors = vector_search(connection, no_answer_vector, TOP_K)
            no_answer_text = fulltext_search(connection, NO_ANSWER_QUESTION, TOP_K)
            no_answer_hybrid = reciprocal_rank_fusion(no_answer_vectors, no_answer_text)[:TOP_K]

        print("\nОТДЕЛЬНЫЙ ТЕСТ ВОПРОСА БЕЗ ОТВЕТА В КОРПУСЕ")
        print(NO_ANSWER_QUESTION)
        print(f"Vector candidates: {len(no_answer_vectors)}")
        print(f"Full-text candidates: {len(no_answer_text)}")
        print(f"Hybrid candidates: {len(no_answer_hybrid)}")
        print(
            "Примечание: этот вопрос не включён в Precision/Recall/MRR/nDCG, "
            "поскольку для него нет релевантного документа. Отказ LLM проверяется отдельно."
        )
        print(
            "\nОграничение эксперимента: в корпусе только два документа и два фрагмента "
        )
        return 0

    except (psycopg2.Error, RuntimeError, OSError) as error:
        print(f"Ошибка оценки поиска: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
