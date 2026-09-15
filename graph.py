import json
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END

from graph_tools import analyze_dataset, search_papers, verify_models

GRAPH_INSTANCE = None

class AoIState(TypedDict, total=False):
    user_query: str
    dataset_path: str

    dataset_analysis: dict
    normalized_task: str
    papers: list
    models: list[dict]
    verified_models: list[dict]
    recommendations: str


def build_recommendation_graph(llm: BaseChatModel):
    def analyze_dataset_node(state: AoIState) -> dict:
        print("\n analyze_dataset step")
        return {
            "dataset_analysis": analyze_dataset.invoke({
                "dataset_path": state["dataset_path"]})
        }

    def normalize_task_node(state: AoIState) -> dict:
        print("\n normalize_task step")
        response = llm.invoke([
            (
                "system",
                """Normalize the user's request into a concise machine-learning task.

Return only the task description.
Do not include Markdown.
Do not include the dataset filesystem path.
Do not provide recommendations.
Do not prefix the response with labels such as "Task:"."""
            ),
            (
                "user",
                f"""User request:
{state["user_query"]}

Dataset information:
{state["dataset_analysis"]}"""
            ),
        ])

        return {"normalized_task": response.content.strip()}

    def search_papers_node(state: AoIState) -> dict:
        print("\n search_papers step")
        return {
            "papers": search_papers.invoke({
                "query": state["normalized_task"],
                "max_results": 10,
            })
        }

    def extract_models_node(state: AoIState) -> dict:
        print("\n extract_models step")
        response = llm.invoke([
            (
                "system",
                """Extract machine learning models mentioned in the supplied papers.

Return ONLY valid JSON as an array of objects with exactly these keys:

[
  {
    "model": "Random Forest",
    "paper": "Paper title",
    "paper_citations": 123
  }
]

Rules:
- "model" is the machine learning model name.
- "paper" is the title of the paper mentioning the model.
- "paper_citations" is the citation count for that paper.
- Include one object per model/paper combination.
- Return [] if no models are found.
- Do not return Markdown or explanatory text."""
            ),
            (
                "user",
                str(state["papers"])
            ),
        ])
        content = response.content.strip()
        if content.startswith("```"):
            content = content.removeprefix("```json").removeprefix("```")
            content = content.removesuffix("```").strip()
        try:
            models = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Model extraction returned invalid JSON: {content}"
            ) from e

        try:
            models = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Model extraction returned invalid JSON: {content}"
            ) from e

        if not isinstance(models, list):
            raise ValueError("Expected model extraction to return a list.")

        return {"models": models}

    def verify_models_node(state: AoIState) -> dict:
        print("\n verify_models step")
        return {
            "verified_models": verify_models.invoke({
                "models": state["models"]
            })
        }

    def rank_models_node(state: AoIState) -> dict:
        print("\n rank_models step")
        response = llm.invoke([
            (
                "system",
                """Rank the candidate machine-learning models for the task.
Consider:
- suitability for the dataset type
- dataset size
- classification characteristics
- expected predictive performance
- interpretability
- computational complexity
- Hugging Face popularity as a secondary signal

For EACH recommended model, include:

1. Model name
2. Why it is appropriate for the task
3. Paper title
4. Paper citation count
5. Hugging Face URL if available
6. Hugging Face downloads if hf_url available
7. Hugging Face likes if hf_url available

Use this format:

## 1. Model Name

Why: Brief explanation of why the model is suitable.

Paper: Paper title  
 - citations: 123

Hugging Face: "https://huggingface.co/..." OR "Not Available"
 - Downloads: 12345  
 - Likes:** 42

If a paper URL or Hugging Face URL is unavailable, write "Not available".

Rank the models from most appropriate to least appropriate.

Keep the explanations concise. Do not invent paper or Hugging Face information. Only use information supplied in the candidate models."""
            ),
            (
                "user",
                f"""Task:
    {state["normalized_task"]}

    Dataset analysis:
    {state["dataset_analysis"]}

    Verified candidate models:
    {state["verified_models"]}"""
            ),
        ])

        return {"recommendations": response.content}


    builder = StateGraph(AoIState)

    builder.add_node("analyze_dataset", analyze_dataset_node)
    builder.add_node("normalize_task", normalize_task_node)
    builder.add_node("search_papers", search_papers_node)
    builder.add_node("extract_models", extract_models_node)
    builder.add_node("verify_models", verify_models_node)
    builder.add_node("rank_models", rank_models_node)

    builder.add_edge(START, "analyze_dataset")
    builder.add_edge("analyze_dataset", "normalize_task")
    builder.add_edge("normalize_task", "search_papers")
    builder.add_edge("search_papers", "extract_models")
    builder.add_edge("extract_models", "verify_models")
    builder.add_edge("verify_models", "rank_models")
    builder.add_edge("rank_models", END)

    graph = builder.compile()
    global GRAPH_INSTANCE
    GRAPH_INSTANCE = graph
    return graph

def run_recommendation_graph(
    user_query: str,
    dataset_path: str,
) -> dict:
    graph = GRAPH_INSTANCE
    assert graph is not None, "build_recommendation_graph() must \
        be called before run_recommendation_graph()"
    recommendation = graph.invoke({
        "user_query": user_query,
        "dataset_path": dataset_path,
    })
    print("\nfinished!")
    return recommendation