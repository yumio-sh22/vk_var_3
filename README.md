# vk_var_3
## Практическая работа № 2 — вариант 3: RAG + векторная БД

Gрототип поиска по корпоративным документам PDF и DOCX. Система извлекает текст, разбивает его на фрагменты, создаёт embeddings, находит релевантные фрагменты и формирует ответ с указанием источника.

## Как работает

1. `ingest.py` извлекает текст из PDF (`pypdf`) и DOCX (`python-docx`), разбивает его на фрагменты и создаёт embeddings.
2. PostgreSQL с расширением pgvector хранит документы, фрагменты и векторы размерностью 768.
3. `search.py` выполняет векторный поиск (HNSW), полнотекстовый поиск (GIN) и объединяет результаты методом RRF.
4. `generate.py` передаёт найденный контекст локальной модели `gemma4:e4b` через Ollama и выводит ответ с источниками.
5. `evaluate.py` запускает оценку поиска на размеченных контрольных вопросах.

## Технологии

Python, Docker Compose, PostgreSQL, pgvector, Ollama, `nomic-embed-text`, `gemma4:e4b`, HNSW, GIN, RRF.

## Структура проекта

- `documents/` — исходные PDF- и DOCX-документы;
- `sql/01_init.sql` — расширение, таблицы и индексы;
- `docker-compose.yml` — PostgreSQL и Ollama;
- `ingest.py` — загрузка и подготовка документов;
- `search.py` — векторный, полнотекстовый и гибридный поиск;
- `generate.py` — генерация ответа по найденному контексту;
- `evaluate.py` — контрольная оценка поиска;
- `requirements.txt` — Python-зависимости.

## Запуск (Windows PowerShell)

```powershell
docker compose up -d
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull gemma4:e4b

py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt

.\.venv\Scripts\python.exe .\ingest.py
.\.venv\Scripts\python.exe .\search.py "Кто проверяет обоснование запроса доступа?"
.\.venv\Scripts\python.exe .\generate.py "Кто проверяет обоснование запроса доступа и где фиксируется решение?"
.\.venv\Scripts\python.exe .\evaluate.py
```

## Результат и ограничения

В проверенном корпусе находятся два документа (`regulation.pdf` и `instruction.docx`), два фрагмента и два embedding-вектора размерностью 768. Проверены загрузка документов, поиск, генерация ответа с источником и отказ от ответа при отсутствии нужных сведений.
