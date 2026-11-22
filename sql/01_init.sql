CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    doc_id          BIGSERIAL PRIMARY KEY,
    source_uri      TEXT NOT NULL UNIQUE,
    title           TEXT NOT NULL,
    file_type       TEXT NOT NULL CHECK (file_type IN ('pdf', 'docx')),
    version         TEXT,
    checksum_sha256 TEXT NOT NULL,
    department      TEXT,
    allowed_groups  TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    modified_at     TIMESTAMPTZ,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_active       BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id         BIGSERIAL PRIMARY KEY,
    doc_id           BIGINT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_no         INTEGER NOT NULL,
    page_no          INTEGER,
    section_title    TEXT,
    content          TEXT NOT NULL,
    embedding        VECTOR(768),
    embedding_model  TEXT,
    search_vector    TSVECTOR GENERATED ALWAYS AS
                     (to_tsvector('simple', COALESCE(content, ''))) STORED,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (doc_id, chunk_no)
);

-- Approximate nearest-neighbor search over document chunk embeddings.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Full-text retrieval for exact terms, names, numbers and document wording.
CREATE INDEX IF NOT EXISTS chunks_search_vector_gin_idx
    ON chunks USING GIN (search_vector);

CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS documents_department_idx ON documents (department);
CREATE INDEX IF NOT EXISTS documents_active_idx ON documents (is_active);
