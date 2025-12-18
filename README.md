# MyGov Search — Architecture & Flow

This README describes how the search and suggestion system works in this repository. It contains a Mermaid flowchart you can render with a Mermaid viewer (VS Code Mermaid Preview, GitHub, or an online Mermaid live editor), a concise ASCII flow, file-to-function mapping, and next steps.

```mermaid
flowchart TD
  A[Client Request\n(user)] -->|POST /suggest| B[/suggest endpoint\n(`main.py`) ]
  A -->|POST /search| C[/search endpoint\n(`main.py`)]

  subgraph API["FastAPI (`main.py`)"]
    B --> D[suggest_queries()]
    C --> E[get_embedding()] 
    E --> F[Embeddings Server\n(OpenAI-compatible)]
    C --> G[pg_vector_search()]
  end

  D --> H[get_lexical_data() cache]
  H --> I[load_search_documents() -> parse `search_service_dump.json`]
  I --> J[build_lexical_models() -> prefix / bigram / vocab]

  subgraph Indexer["Indexer (`txtai-search.py`)"]
    K[index_dataset()]
    K --> L[load_mygov_data() -> prepare_docs()]
    L --> M[embed_texts() -> Embeddings Server]
    M --> N[upsert_services() -> Postgres table `mygov_services`]
    K -->|build lexical models| H
  end

  G --> O[Postgres `mygov_services`]
  F --> O
  N --> O

  subgraph Postgres["Postgres DB"]
    O[mygov_services (bn_name, en_name, keywords, embedding float8[])]
  end

  style API fill:#f8f9fa,stroke:#333
  style Indexer fill:#f0f8ff,stroke:#333
  style Postgres fill:#fff7e6,stroke:#333
```

**ASCII Quick Flow**

- Client → POST `/suggest` (in `main.py`)
  - `/suggest` calls `suggest_queries()` → uses cached `get_lexical_data()`
  - `get_lexical_data()` loads parsed data from `search_service_dump.json` (`load_search_documents()`), builds lexical models (`build_lexical_models()`), and returns prefix/bigram/vocab sets → suggestions returned

- Client → POST `/search` (in `main.py`)
  - `/search` → `get_embedding(query)` calls OpenAI-compatible embeddings API
  - Normalize embedding → call `pg_vector_search(embedding, query, top_k)`
    - If query tokens produce a lexical pattern:
      - SQL: filter rows WHERE `en_name` ILIKE / `bn_name` ILIKE OR `keywords` ILIKE pattern AND compute dot product (unnest embedding) with normalized query → ORDER BY score DESC LIMIT k
    - If lexical filter returns no rows:
      - Fallback: rank entire table by dot product (no ILIKE)
  - Results mapped to `SearchItem` and returned JSON

- Indexing (one-time or periodic) — `txtai-search.py`
  - `index_dataset()`:
    - `load_mygov_data()` reads `search_service_dump.json` (NDJSON or JSON)
    - `prepare_docs()` creates tuples (doc_id, bn_name, en_name, keywords, text)
    - `embed_texts()` calls embeddings server in batches
    - `upsert_services()` inserts rows into Postgres `mygov_services` (embedding float8[])
  - Also builds prefix/next-word/vocab for suggestions (shared shape used by `get_lexical_data()`)

**Mapping to files & functions**

- `main.py`
  - API endpoints: `/suggest` and `/search`
  - Lexical: `load_search_documents()`, `build_lexical_models()`, `get_lexical_data()`, `suggest_queries()`
  - Embedding call: `get_embedding()`
  - Search logic: `pg_vector_search()` (normalizes embedding, creates lexical pattern, queries DB)
  - DB schema helper: `init_schema()` (ensures `mygov_services` exists)

- `txtai-search.py`
  - Data loader & indexer: `load_mygov_data()`, `prepare_docs()`, `index_dataset()`
  - Embedding helper: `embed_texts()`
  - Upsert into DB: `upsert_services()`
  - Local lexical builder used during indexing (keeps `prefix_suggestions`, `next_word`, `vocab_words`)

- `dataset.py`
  - (Optional) Extracts raw rows from MySQL and writes `mygov_data.json`, `Mygovdata.csv`

- Database: table `mygov_services` holds embeddings (`float8[]`) and textual fields used by lexical matching

**Key Implementation Details**

- **Embeddings**: produced by OpenAI-compatible client (`client.embeddings.create`). Results are truncated/padded to `EMBEDDING_DIM`.
- **Query vector**: L2-normalized in `pg_vector_search` so dot-product approximates cosine similarity.
- **Lexical boost**: extract longest token from query → build pattern `%token%` → SQL `ILIKE` filter on `en_name`, `bn_name`, `keywords` to restrict candidates before scoring by vector dot-product.
- **Fallback**: if lexical restriction returns zero rows, runs pure vector ranking over full table.
- **Suggestion engine**: uses prefix maps, bigrams, and vocabulary derived from the dataset text.

**Next Steps & Suggestions**

- **Render diagram**: export the Mermaid diagram to PNG/SVG using a Mermaid renderer or `mmdc` (Mermaid CLI).
- **Add architecture section**: include this README content into a project README or docs.
- **Run indexer**: run `txtai-search.py` to populate `mygov_services` (requires Postgres credentials and embeddings server).

If you want, I can render the Mermaid diagram into `docs/architecture.png` and commit it here. Tell me if you want that (and whether you prefer `png` or `svg`).
