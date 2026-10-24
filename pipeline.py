"""Orchestrate ResearchPilot's search, reading, writing, and review stages."""

from __future__ import annotations

import json
import logging
from typing import Any

from agents import build_scrape_agent, build_search_agent, critic_chain, writer_chain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _final_agent_text(result: dict[str, Any]) -> str:
    """Return the last assistant message, avoiding tool-call messages if possible."""
    messages = result.get("messages", [])
    for message in reversed(messages):
        role = getattr(message, "type", "")
        content = getattr(message, "content", None)
        if role == "ai" and isinstance(content, str) and content.strip():
            return content.strip()
    raise RuntimeError("Agent finished without returning a text response.")


def _validate_topic(topic: str) -> str:
    topic = " ".join(topic.split())
    if not topic:
        raise ValueError("Research topic cannot be empty.")
    if len(topic) > 500:
        raise ValueError("Research topic must be 500 characters or fewer.")
    return topic


def run_research_pipeline(topic: str) -> dict[str, Any]:
    """Run the four-stage research workflow and return its state/results."""
    topic = _validate_topic(topic)
    state: dict[str, Any] = {
        "topic": topic,
        "search_results": "",
        "scraped_content": "",
        "report": "",
        "feedback": "",
        "errors": [],
    }

    logger.info("Step 1/4: searching the web")
    try:
        search_result = build_search_agent().invoke(
            {"messages": [("user", f"Find recent, relevant, reliable sources about: {topic}")]}
        )
        state["search_results"] = _final_agent_text(search_result)
        logger.info("Search stage completed")
    except Exception as exc:
        logger.exception("Search stage failed")
        state["errors"].append({"stage": "search", "error": str(exc)})
        raise RuntimeError(f"ResearchPilot search stage failed: {exc}") from exc

    logger.info("Step 2/4: reading relevant webpages")
    try:
        # Pass the complete result instead of silently truncating it to 800 chars.
        reader_prompt = (
            f"Research topic: {topic}\n\n"
            "Search results follow. Select up to three distinct, relevant URLs and scrape them. "
            "Preserve source URLs and clearly separate evidence by source.\n\n"
            f"{state['search_results']}"
        )
        reader_result = build_scrape_agent().invoke(
            {"messages": [("user", reader_prompt)]}
        )
        state["scraped_content"] = _final_agent_text(reader_result)
        logger.info("Reader stage completed")
    except Exception as exc:
        logger.exception("Reader stage failed")
        state["errors"].append({"stage": "reader", "error": str(exc)})
        # Search results can still provide limited evidence for a report.
        state["scraped_content"] = f"Deeper webpage reading failed: {type(exc).__name__}."

    research_combined = (
        "SEARCH RESULTS\n"
        f"{state['search_results']}\n\n"
        "SCRAPED SOURCE CONTENT\n"
        f"{state['scraped_content']}"
    )
    state["research"] = research_combined

    logger.info("Step 3/4: drafting the report")
    try:
        state["report"] = writer_chain.invoke(
            {"topic": topic, "research": research_combined}
        )
    except Exception as exc:
        logger.exception("Writer stage failed")
        state["errors"].append({"stage": "writer", "error": str(exc)})
        raise RuntimeError(f"ResearchPilot writer stage failed: {exc}") from exc

    logger.info("Step 4/4: reviewing the report")
    try:
        state["feedback"] = critic_chain.invoke(
            {
                "topic": topic,
                "research": research_combined,
                "report": state["report"],
            }
        )
    except Exception as exc:
        logger.exception("Critic stage failed")
        state["errors"].append({"stage": "critic", "error": str(exc)})
        state["feedback"] = f"Critic stage failed ({type(exc).__name__}); report was still generated."

    logger.info("Research pipeline finished")
    return state


if __name__ == "__main__":
    try:
        user_topic = input("Enter a research topic: ")
        result = run_research_pipeline(user_topic)
        print("\n" + "=" * 72 + "\nFINAL REPORT\n" + "=" * 72)
        print(result["report"])
        print("\n" + "=" * 72 + "\nCRITIC FEEDBACK\n" + "=" * 72)
        print(result["feedback"])
        if result["errors"]:
            print("\nPipeline warnings/errors:")
            print(json.dumps(result["errors"], indent=2))
    except (ValueError, RuntimeError) as exc:
        logger.error("Pipeline stopped: %s", exc)
