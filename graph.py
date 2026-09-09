# type: ignore
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END

from graph_tools import analyze_dataset, search_papers

GRAPH_INSTANCE = None

class AoIState(TypedDict, total=False):
    user_query: str
    dataset_path: str

    dataset_analysis: dict
    normalized_task: str
    papers: list
    models: list
    verified_models: list
    recommendations: str


def build_recommendation_graph(llm: BaseChatModel):
    def analyze_dataset_node(state: AoIState) -> dict:
        return {
            "dataset_analysis": analyze_dataset.invoke({
                "dataset_path": state["dataset_path"]})
        }

    def normalize_task_node(state: AoIState) -> dict:
        response = llm.invoke([
            (
                "system",
                "Normalize the user's ML request into a concise task description."
            ),
            (
                "user",
                f"""User request:
{state["user_query"]}

Dataset information:
{state["dataset_analysis"]}"""
            ),
        ])

        return {"normalized_task": response.content}

    def search_papers_node(state: AoIState) -> dict:
        return {
            "papers": search_papers.invoke({
                "query": state["normalized_task"],
                "max_results": 10,
            })
        }

    def extract_models_node(state: AoIState) -> dict:
        response = llm.invoke([
            (
                "system",
                "Extract the machine learning model names mentioned "
                "in the supplied papers. Return only model names."
            ),
            (
                "user",
                str(state["papers"])
            ),
        ])

        return {"models": response.content}

    def verify_models_node(state: AoIState) -> dict:
        return {
            "verified_models": verify_models.invoke({
                "models": state["models"]
            })
        }

    def rank_models_node(state: AoIState) -> dict:
        response = llm.invoke([
            (
                "system",
                "Rank the candidate models for the requested task. "
                "Give a concise reason for each recommendation."
            ),
            (
                "user",
                f"""Task:
{state["normalized_task"]}

Candidate models:
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
) -> AoIState:
    graph = GRAPH_INSTANCE
    assert graph is not None, "build_recommendation_graph() must \
        be called before run_recommendation_graph()"
    return graph.invoke({
        "user_query": user_query,
        "dataset_path": dataset_path,
    })