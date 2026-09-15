from typing import Any
import requests
from langchain_core.tools import tool
from graph import run_recommendation_graph

@tool(return_direct=True)
def recommend_model(user_query: str, dataset_path: str) -> str:
    """Research and recommend ML models for a user's task.

    Args:
        user_query (str): The user's original query
        dataset_path (str): the path to the dataset

    Returns:
        dict[str, Any]: a dictionary containing "task", "recommendations", and "models"
    """
    print("Using tool recommend_model")
    state = run_recommendation_graph(
        user_query=user_query,
        dataset_path=dataset_path,
    )

    return state["recommendations"]

@tool
def search_ndp_catalog(query: str, max_results:int = 10) -> dict[str, Any]:
    """
    Search datasets in the NDP catalog.

    Parameters:
        query (str): query e.g. "climate"
        max_results (int): max results to return (Defaults to 10)

    Returns:
        dict[str, Any]: a dict of one of the following formats:
            - {"ok": True, "count":int, "datasets":dict}
            - {"ok":False, "error":str} 

    """
    print("Using tool search_ndp_catalog")
    catalog_url = "https://nationaldataplatform.org/catalog"
    try:
        response = requests.get(
            f"{catalog_url}/api/3/action/package_search",
            params={"q": query, "rows": max_results},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        if not data.get("success"):
            return {
                "ok": False,
                "error": data.get("error", "NDP catalog search failed"),
            }
        result = data["result"]
        datasets = [
            {
                "id": dataset.get("id"),
                "name": dataset.get("name"),
                "title": dataset.get("title"),
                "description": dataset.get("notes"),
                "url": dataset.get("url"),
                "organization": (
                    dataset.get("organization") or {}
                ).get("title"),
                "tags": [
                    tag.get("name")
                    for tag in dataset.get("tags", [])
                ],
                "resources": [
                    {
                        "name": resource.get("name"),
                        "format": resource.get("format"),
                        "url": resource.get("url"),
                    }
                    for resource in dataset.get("resources", [])
                ],
            }
            for dataset in result.get("results", [])
        ]

        return {
            "ok": True,
            "count": result.get("count", 0),
            "datasets": datasets,
        }

    except requests.RequestException as e:
        return {
            "ok": False,
            "error": str(e),
        }
    
TOOLS = [
    recommend_model,
    search_ndp_catalog
]