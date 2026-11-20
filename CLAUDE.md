# rag-example

PDF RAG chatbot (LangChain + FastAPI + FAISS) copied from company-demos `projects/llms/general/langchain-rag`.
Targets the **riley-demo** workspace only: project "RAG Example" `3319c41a-3e05-432e-a908-ef07be2c98e5`,
pipeline `production` `041cad98-0246-41c3-a8ac-db670edab6d1`. Ids and the key live in `.env` (gitignored,
never print it). Run with `uv run`; Python 3.10 (matches `openlayer.json` runtime). Keep `numpy<2` while
faiss-cpu is 1.8.0. Use cheap models only (gpt-4o-mini default via `OPENAI_MODEL`).
