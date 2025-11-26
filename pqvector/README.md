# MyGov Semantic Search / Recommendation API

This project builds a semantic search / recommendation API for MyGov services using:

- Text embeddings (from LM Studio)
- PostgreSQL with pgvector for vector storage and ANN search
- FastAPI as the HTTP API layer

High-level flow:

1. Load service data from `dataset/mygov_data.json`
2. Generate embeddings using an embedding model served by **LM Studio**
3. Store vectors in PostgreSQL (`mgov` database) with **pgvector**
4. Expose a FastAPI endpoint that performs cosine similarity search using pgvector

---

## Tech Stack

- **LM Studio** (OpenAI-compatible embeddings server)
- **Python 3.10+**
- **FastAPI** + **Uvicorn**
- **PostgreSQL 18** + **pgvector**  
  Docker image: `pgvector/pgvector:pg18`
- **psycopg** (PostgreSQL client)
- **pgvector** (Python extension to work with Postgres `vector` type)

---

## Project Structure

```text
project-root/
├─ dataset/
│  └─ mygov_data.json        # MyGov service data
├─ src/
│  ├─ embeddings.py          # data → embeddings → Postgres (indexer)
│  ├─ search_service.py      # query → embedding → pgvector search
│  └─ main.py                # FastAPI app
├─ requirements.txt
└─ README.md
