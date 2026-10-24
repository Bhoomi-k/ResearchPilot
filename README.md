# ResearchPilot — first reliability upgrade

This version preserves the four-stage workflow: search agent, reader agent, writer chain, and critic chain.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and set `GROQ_API_KEY` and `TAVILY_API_KEY`.
4. Run: `python pipeline.py`

## Files

- `tools.py`: Tavily search and guarded HTML scraping tools.
- `agents.py`: search/reader agents and writer/critic chains.
- `pipeline.py`: orchestration, input validation, logging, and error handling.

## Important notes

- This is a first reliability improvement, not a complete production security review.
- URL checks block obvious localhost/private-IP destinations, but production deployments should enforce outbound network restrictions and validate DNS resolution to mitigate SSRF and DNS-rebinding risks.
- The scraper supports HTML/text pages, not JavaScript-rendered websites or PDF extraction.
- The writer is instructed to cite only supplied URLs, but citations still require automated/source-level verification before being treated as trustworthy.
- RAG indexing, chunking, embeddings, persistent vector storage, hybrid retrieval, and evaluation are not included yet.
