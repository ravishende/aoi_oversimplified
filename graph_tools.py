# pylint: disable=typecheck
from __future__ import annotations
from copy import deepcopy
from huggingface_hub import HfApi
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

@tool
def verify_models(models: list[dict]) -> list[dict]:
    """
    Find the best Hugging Face match for each extracted ML model.

    Preserves the original model dictionary and adds:
      - hf_link
      - hf_downloads
      - hf_likes
    """

    api = HfApi()
    verified_models = []

    for model in models:
        result = deepcopy(model)

        model_name = str(model.get("model", "")).strip()

        # Defaults when no HF match is found.
        result["hf_link"] = None
        result["hf_downloads"] = 0
        result["hf_likes"] = 0

        if not model_name:
            verified_models.append(result)
            continue

        try:
            print(f"Checking Hugging Face for: {model_name}")

            matches = list(
                api.list_models(
                    search=model_name,
                    sort="downloads",
                    limit=1,
                )
            )

            if matches:
                match = matches[0]

                result["hf_link"] = (
                    f"https://huggingface.co/{match.id}"
                )
                result["hf_downloads"] = (
                    getattr(match, "downloads", 0) or 0
                )
                result["hf_likes"] = (
                    getattr(match, "likes", 0) or 0
                )

        except Exception as e:
            # Keep the model and its paper information even if
            # Hugging Face lookup fails.
            print(
                f"Hugging Face lookup failed for "
                f"{model_name}: {e}"
            )

        verified_models.append(result)

    # Put models with stronger HF presence first, while retaining
    # paper citations as a secondary signal.
    verified_models.sort(
        key=lambda m: (
            m.get("hf_link") is not None,
            m.get("hf_downloads", 0),
            m.get("paper_citations", 0),
        ),
        reverse=True,
    )

    return verified_models

TOOLS = [
    search_papers,
    analyze_dataset,
    verify_models
]