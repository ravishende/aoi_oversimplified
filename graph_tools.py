# pylint: disable=typecheck
from __future__ import annotations
from langchain_core.tools import tool
import requests

OPENALEX_BASE_URL = "https://api.openalex.org/works"


@tool
def analyze_dataset(dataset_path: str) -> dict:
    """Analyze a dataset and return basic metadata."""
    # Hard-coded dataset analysis -- would replace with something intelligent
    if "iris" in dataset_path:
        return {
            "path": dataset_path,
            "modality": "tabular",
            "task": "tabular classification"
        }
    return {
        "path": dataset_path,
        "modality": "image",
        "task": "image classification",
    }


@tool
def search_papers(
    query: str,
    max_results: int = 10,
) -> list:
    """
    Search OpenAlex API (free, no API key required).

    OpenAlex is an open catalog of scholarly works, replacing Microsoft Academic.
    Returns papers with citation counts.

    API Docs: https://docs.openalex.org/
    """

    params = {
        "search": query,
        "per-page": max_results,
        "sort": "relevance_score:desc",
    }

    try:
        response = requests.get(
            OPENALEX_BASE_URL,
            params=params,
            timeout=15,
        )
        response.raise_for_status()
        papers = []
        for work in response.json().get("results", []):
            authors = [
                authorship.get("author", {}).get("display_name")
                for authorship in work.get("authorships", [])
                if authorship.get("author", {}).get("display_name")
            ]
            primary_location = work.get("primary_location") or {}
            source = primary_location.get("source") or {}
            url = work.get("doi") or work.get("id", "")

            if not url.startswith("https://doi.org/"):
                url = primary_location.get("landing_page_url") or url

            papers.append({
                "source": "openalex",
                "title": work.get("title", ""),
                "abstract": work.get("abstract", "") or "",
                "year": work.get("publication_year"),
                "citation_count": work.get("cited_by_count", 0),
                "authors": authors[:10],
                "paper_id": work.get("id", ""),
                "url": url,
                "venue": source.get("display_name", ""),
                "publication_date": work.get("publication_date", ""),
                "openalex_id": work.get("id", ""),
            })

        return papers

    except requests.RequestException as e:
        print(f"OpenAlex search error: {e}")
        return []

TOOLS = [
    search_papers
]