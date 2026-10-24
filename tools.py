"""Tools used by ResearchPilot's search and reader agents."""

from __future__ import annotations

import ipaddress
import json
import os
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import tool
from tavily import TavilyClient

load_dotenv()

USER_AGENT = (
    "Mozilla/5.0 (compatible; ResearchPilot/1.0; "
    "+https://github.com/Bhoomi-k/ResearchPilot)"
)
REQUEST_TIMEOUT_SECONDS = 12
MAX_PAGE_CHARS = 12_000


def _get_tavily_client() -> TavilyClient:
    """Create the Tavily client only when search is requested."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is missing. Add it to your .env file."
        )
    return TavilyClient(api_key=api_key)


def _is_allowed_public_url(url: str) -> tuple[bool, str]:
    """Basic URL checks to block obvious local/private-network targets.

    Production deployments should also enforce egress/network restrictions and
    validate DNS-resolved IPs at connection time to mitigate DNS rebinding.
    """
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return False, "The URL is malformed."

    if parsed.scheme not in {"http", "https"}:
        return False, "Only http:// and https:// URLs are allowed."
    if not parsed.hostname:
        return False, "The URL must include a hostname."
    if parsed.username or parsed.password:
        return False, "URLs containing embedded credentials are not allowed."

    host = parsed.hostname.lower().rstrip(".")
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        return False, "Local hostnames are not allowed."

    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        # Hostnames are allowed; a production service should additionally
        # validate resolved IPs and restrict outbound network access.
        return True, ""

    if not ip.is_global:
        return False, "Private, loopback, link-local, and reserved IPs are not allowed."
    return True, ""


@tool
def websearch(query: str) -> str:
    """Search the live web and return up to five results as structured JSON.

    Each result includes a title, URL, snippet, and search score when available.
    """
    query = query.strip()
    if not query:
        return json.dumps({"results": [], "error": "Search query is empty."})

    try:
        response = _get_tavily_client().search(
            query=query,
            max_results=5,
            search_depth="advanced",
            include_answer=False,
        )
        results = []
        for item in response.get("results", []):
            url = item.get("url")
            if not url:
                continue
            results.append(
                {
                    "title": item.get("title", "Untitled source"),
                    "url": url,
                    "snippet": item.get("content", ""),
                    "score": item.get("score"),
                }
            )
        return json.dumps({"query": query, "results": results}, ensure_ascii=False)
    except Exception as exc:
        # Return a tool-readable error rather than a traceback or secret.
        return json.dumps(
            {"query": query, "results": [], "error": f"Web search failed: {type(exc).__name__}"}
        )


@tool
def webscrape(url: str) -> str:
    """Fetch a public HTML webpage and return cleaned text plus its final URL."""
    url = url.strip()
    allowed, reason = _is_allowed_public_url(url)
    if not allowed:
        return f"Scraping blocked: {reason}"

    try:
        response = requests.get(
            url,
            timeout=(4, REQUEST_TIMEOUT_SECONDS),
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
            allow_redirects=True,
        )
        response.raise_for_status()

        # Re-check the final URL after redirects.
        allowed_final, final_reason = _is_allowed_public_url(response.url)
        if not allowed_final:
            return f"Scraping blocked after redirect: {final_reason}"

        content_type = response.headers.get("Content-Type", "").lower()
        if content_type and "html" not in content_type and "text/plain" not in content_type:
            return f"Unsupported content type: {content_type}. Only HTML/text pages are supported."

        soup = BeautifulSoup(response.text, "lxml")
        for tag in soup(
            ["script", "style", "nav", "footer", "header", "aside", "noscript", "svg", "form"]
        ):
            tag.decompose()

        # Prefer the main article when the page exposes one.
        main = soup.find("article") or soup.find("main") or soup.body or soup
        text = main.get_text(separator=" ", strip=True)
        text = " ".join(text.split())

        if not text:
            return f"No readable text was found at {response.url}"

        truncated = len(text) > MAX_PAGE_CHARS
        text = text[:MAX_PAGE_CHARS]
        suffix = "\n[Page text truncated at configured character limit.]" if truncated else ""
        return (
            f"Original URL: {url}\n"
            f"Final URL: {response.url}\n"
            f"Page title: {soup.title.get_text(strip=True) if soup.title else 'Unknown'}\n"
            f"Page content:\n{text}{suffix}"
        )
    except requests.Timeout:
        return f"Scraping timed out for URL: {url}"
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        return f"HTTP error while scraping URL (status {status}): {url}"
    except requests.RequestException as exc:
        return f"Network error while scraping URL ({type(exc).__name__}): {url}"
    except Exception as exc:
        return f"Could not extract page content ({type(exc).__name__}): {url}"
