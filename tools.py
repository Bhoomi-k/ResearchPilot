from langchain.tools import tool
import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
from rich import print

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

#creating tools for websearch
@tool
def websearch(query: str) -> str:
    """Search the web for the recent and the reliable information based on the query and return the Titles, URLs, and Snippets of the top 5 results."""
    response = tavily.search(query, max_results=5)  #return the list of dictionary having title, url, content, score, and raw_content.

    out = []

    for result in response['results']:
        out.append(
            f"Title: {result['title']}\nURL: {result['url']}\nSnippet: {result['content']}\n"
        )

    return "\n-----\n".join(out)

# print(websearch.invoke("what is the latest news on protest in delhi?"))

#creating tools for webscraping
# @tool
# def webscrape(url: str) -> str:
#     """Scrape the web page at the given URL and return the clean text content of the page for deeper reading."""
#     try:
#         resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
#         soup = BeautifulSoup(resp.content, "html.parser")
#         for tag in soup(["script", "style", "nav", "footer"]):
#             tag.decompose()
#         return soup.get_text(separator=" ", strip=True)[:3000]  # Limit to first 3000 characters and strip=True removes leading and trailing whitespace from the extracted text fragments.
#     except Exception as e:
#         return f"Error occurred while scraping the page: {e}"

@tool
def webscrape(url: str) -> str:
    """Scrape a webpage URL and return its clean text content."""

    try:
        response = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/130.0.0.0 Safari/537.36"
                )
            },
            allow_redirects=True,
        )

        if response.status_code != 200:
            return (
                f"Failed to scrape URL: {url}\n"
                f"HTTP status: {response.status_code}"
            )

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup([
            "script", "style", "nav", "footer",
            "header", "aside", "noscript"
        ]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)

        if not text:
            return f"The page loaded but no readable text was found: {url}"

        return (
            f"Original URL: {url}\n"
            f"Final URL: {response.url}\n"
            f"Page content:\n{text[:5000]}"
        )

    except requests.RequestException as exc:
        return f"Request failed for {url}: {exc}"