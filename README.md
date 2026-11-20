# RAG Example (PDF RAG chatbot, LangChain + FastAPI)

Copy of `company-demos/projects/llms/general/langchain-rag`, wired to Riley's **riley-demo** Openlayer workspace.

| | |
|---|---|
| Workspace | `riley-demo` (`559fe165-099a-400a-9d58-14013d6f0dd2`) |
| Project | RAG Example (`3319c41a-3e05-432e-a908-ef07be2c98e5`) |
| Inference pipeline | `production` (`041cad98-0246-41c3-a8ac-db670edab6d1`) |

`.env` (gitignored) holds the riley-demo API key and these ids; it overrides shell variables. Run with uv:

```bash
uv sync
uv run python openlayer_monitoring.py        # one traced Q&A into the production pipeline
uv run python run_server.py                  # web app on http://localhost:8000
```

Changes from the template: `.env` targets riley-demo; `load_dotenv(override=True)`; the chat model is pinned
to `gpt-4o-mini` (override with `OPENAI_MODEL`) because LangChain's default `gpt-3.5-turbo` is retired;
`numpy<2` because faiss-cpu 1.8.0 can't import under numpy 2; a `pyproject.toml` for uv (Python 3.10, the
`python_3_10` runtime in `openlayer.json`).

### Development mode (GitHub-linked)

The Openlayer project is linked to `riley-ol/rag-example` (branch `main`, root `./`). Every push runs
`openlayer_run.py` on the golden set in Openlayer (`modelType: full`; the workspace `OPENAI_API_KEY` secret is
injected) and posts an **Openlayer** check run on the commit or PR. Tests (`tests.json`):

- Ensure all outputs are valid JSON
- Maximum response latency of 5 seconds (per row, < 6000 ms)
- Answers match the golden answers: `answerCorrectness >= 0.7` against `ground_truth` in `dataset.json`

---

## Original template README

This template shows how Openlayer can be used with a project built with **LangChain** and **FastAPI** to create a RAG (Retrieval-Augmented Generation) chatbot that answers questions about uploaded PDFs. It features Openlayer's:

- **development** mode: used to test your AI system automatically using CI / CD.
- **monitoring** mode: used to test real-time requests to your AI system in production.

## How to use this template

> Prerequisite: an [Openlayer account](https://app.openlayer.com/).

### Monitoring

1. Install the requirements and add your env vars from your [Openlayer project](https://app.openlayer.com/):

```bash
pip install -r requirements.txt
cp .env.example .env # copy your env vars over
```

2. Run the app, which serves the chatbot on `localhost:8000`.

```bash
# Option 1: Use the launcher script (easiest)
python run_server.py

# Option 2: Run with uvicorn from the project root directory
uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload

# Option 3: Navigate to the app directory and run directly
cd app
python server.py

# Option 4: Using uvicorn from the app directory
cd app
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Navigate to `localhost:8000`, upload a PDF document, and start asking questions about its content. You should see these interactions appear in the monitoring mode of your [Openlayer project](https://app.openlayer.com/).

You can also visit `localhost:8000/docs` to see the automatically generated API documentation.

Alternatively, you can run the [`openlayer_monitoring.py`](/python/llms/langchain-rag/openlayer_monitoring.py) script to simulate running your system in production, to stream your system's data and metadata to Openlayer.

### Development

1. Create a GitHub repo that is a clone of this template.

2. Navigate to the [Openlayer platform](https://app.openlayer.com/) and create a project using your repo.

Now, every commit you push to the repo will be automatically tested by Openlayer.

## How it works

This RAG chatbot uses:
- **FastAPI** for the modern, fast web framework with automatic API documentation
- **LangChain** for document processing and RAG implementation
- **FAISS** for vector storage and similarity search
- **OpenAI embeddings** for document chunking and embedding
- **OpenAI Chat models** for generating responses

Passing in Openlayer's callback handler from [Openlayer's Python SDK](https://www.openlayer.com/docs/api-reference/sdk/libraries/python) enables tracing for every chat completion call and retrieval operation. This is done in [`app/model/pdf_chatbot.py`](/python/llms/langchain-rag/app/model/pdf_chatbot.py), where our RAG chatbot is defined.

### Monitoring

All production requests that go through the code that uses the `OpenlayerHandler` are automatically streamed to the Openlayer platform, where our tests and alerts are defined. You just need to add your Openlayer API key and inference pipeline id as environment variables.

### Development

The [`openlayer.json`](/python/llms/langchain-rag/openlayer.json) config file and the [`openlayer_run.py`](/python/llms/langchain-rag/openlayer_run.py) script together allow Openlayer to call your AI system and discover your datasets.

With this, every new commit you push is run against your tests on Openlayer in CI / CD. You can define your tests in the app and update them as your system evolves.

Note that using the `OpenlayerHandler` enables tracing, tokens, and cost tracking in development mode as well.

## Features

- Upload PDF documents or use provided samples
- Interactive chat interface for asking questions
- RAG-based responses with source citations
- Session management for multiple concurrent users
- Real-time monitoring with Openlayer integration
- Automatic document chunking and vector embedding
- Modern FastAPI backend with automatic API documentation
- Async support for better performance
- Type hints throughout for better code quality

## API Endpoints

- `GET /`: Upload page
- `POST /upload-file`: Upload PDF file
- `POST /upload-sample`: Use sample PDF
- `POST /chat`: Chat with the PDF
- `GET /uploads/{filename}`: Download uploaded files
- `GET /health`: Health check endpoint
- `GET /docs`: Interactive API documentation (Swagger UI)
- `GET /redoc`: Alternative API documentation