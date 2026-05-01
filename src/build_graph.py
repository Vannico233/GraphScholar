import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import networkx as nx
from networkx.readwrite import json_graph


def normalize_method_name(tag: str) -> Optional[str]:
    lowered = tag.lower()
    if "contrastive" in lowered:
        return "Contrastive Learning"
    if "prompt" in lowered:
        return "Graph Prompting"
    if "pretrain" in lowered or "pre-train" in lowered:
        return "Graph Pretraining"
    if "knowledge graph" in lowered:
        return "Knowledge-Graph Retrieval"
    if "summarization" in lowered:
        return "Community Summarization"
    if "masked autoencoder" in lowered:
        return "Masked Autoencoder"
    if "retrieval" in lowered:
        return "Graph Retrieval"
    return None


def normalize_topic_name(name: str) -> str:
    return str(name).strip()


def relation_priority(relation_types: list[str]) -> str:
    priority = [
        "survey_of",
        "evaluates",
        "applies_to",
        "shared_task",
        "shared_dataset",
        "shared_application",
        "shared_topic",
    ]
    for relation in priority:
        if relation in relation_types:
            return relation
    return relation_types[0] if relation_types else "related_to"


def add_typed_node(graph: nx.Graph, node_id: str, node_type: str, **attrs: Any) -> None:
    graph.add_node(node_id, node_type=node_type, **attrs)


def build_paper_graph(papers: list[dict[str, Any]]) -> nx.Graph:
    graph = nx.Graph()
    paper_nodes = []

    for paper in papers:
        paper_id = f"paper::{paper['file_name']}"
        add_typed_node(
            graph,
            paper_id,
            "paper",
            title=paper.get("title", ""),
            file_name=paper.get("file_name", ""),
            category=paper.get("category", ""),
            paper_type=paper.get("paper_type", ""),
            abstract=paper.get("abstract", ""),
            tags=paper.get("tags", []),
            tasks=paper.get("tasks", []),
            applications=paper.get("applications", []),
            datasets=paper.get("datasets", []),
        )
        paper_nodes.append((paper_id, paper))

        category = str(paper.get("category", "")).strip()
        if category:
            topic_id = f"topic::{category}"
            add_typed_node(graph, topic_id, "topic", name=category)
            graph.add_edge(paper_id, topic_id, edge_type="paper-topic")

        for tag in paper.get("tags", []):
            tag_name = str(tag).strip()
            if not tag_name:
                continue
            topic_id = f"topic::{tag_name}"
            add_typed_node(graph, topic_id, "topic", name=tag_name)
            graph.add_edge(paper_id, topic_id, edge_type="paper-topic")

            method_name = normalize_method_name(tag_name)
            if method_name:
                method_id = f"method::{method_name}"
                add_typed_node(graph, method_id, "method", name=method_name)
                graph.add_edge(paper_id, method_id, edge_type="paper-method")

        for task in paper.get("tasks", []):
            task_name = normalize_topic_name(task)
            if not task_name:
                continue
            task_id = f"task::{task_name}"
            add_typed_node(graph, task_id, "task", name=task_name)
            graph.add_edge(paper_id, task_id, edge_type="paper-task")

        for application in paper.get("applications", []):
            application_name = normalize_topic_name(application)
            if not application_name:
                continue
            application_id = f"application::{application_name}"
            add_typed_node(graph, application_id, "application", name=application_name)
            graph.add_edge(paper_id, application_id, edge_type="paper-application")

        for dataset in paper.get("datasets", []):
            dataset_name = normalize_topic_name(dataset)
            if not dataset_name:
                continue
            dataset_id = f"dataset::{dataset_name}"
            add_typed_node(graph, dataset_id, "dataset", name=dataset_name)
            graph.add_edge(paper_id, dataset_id, edge_type="paper-dataset")

    for i in range(len(paper_nodes)):
        paper_id_a, paper_a = paper_nodes[i]
        for j in range(i + 1, len(paper_nodes)):
            paper_id_b, paper_b = paper_nodes[j]
            shared_tasks = sorted(set(paper_a.get("tasks", [])) & set(paper_b.get("tasks", [])))
            shared_datasets = sorted(set(paper_a.get("datasets", [])) & set(paper_b.get("datasets", [])))
            shared_applications = sorted(set(paper_a.get("applications", [])) & set(paper_b.get("applications", [])))
            shared_topics = sorted(
                ({
                    str(paper_a.get("category", "")).strip(),
                    *[str(tag).strip() for tag in paper_a.get("tags", [])],
                })
                & ({
                    str(paper_b.get("category", "")).strip(),
                    *[str(tag).strip() for tag in paper_b.get("tags", [])],
                })
            )
            relation_types = []
            type_a = str(paper_a.get("paper_type", "")).lower()
            type_b = str(paper_b.get("paper_type", "")).lower()

            if "survey" in {type_a, type_b}:
                relation_types.append("survey_of")
            if {"benchmark", "evaluation"} & {type_a, type_b}:
                relation_types.append("evaluates")
            if "application" in {type_a, type_b}:
                relation_types.append("applies_to")
            if shared_tasks:
                relation_types.append("shared_task")
            if shared_datasets:
                relation_types.append("shared_dataset")
            if shared_applications:
                relation_types.append("shared_application")
            if shared_topics:
                relation_types.append("shared_topic")

            if relation_types:
                graph.add_edge(
                    paper_id_a,
                    paper_id_b,
                    edge_type="paper-paper",
                    relation_type=relation_priority(relation_types),
                    relation_types=relation_types,
                    shared_tasks=shared_tasks,
                    shared_datasets=shared_datasets,
                    shared_applications=shared_applications,
                    shared_topics=shared_topics,
                )

    return graph


def build_paper_graph_from_json(data_path, save_path=None) -> nx.Graph:
    papers = json.loads(Path(data_path).read_text(encoding="utf-8"))
    graph = build_paper_graph(papers)
    if save_path is not None:
        try:
            save_graph_json(graph, save_path)
        except PermissionError:
            pass
    return graph


def save_graph_json(graph: nx.Graph, save_path) -> Path:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json_graph.node_link_data(graph)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def get_graph_stats(graph: nx.Graph) -> dict[str, Any]:
    node_counter = Counter()
    for _, attrs in graph.nodes(data=True):
        node_counter[attrs.get("node_type", "unknown")] += 1

    edge_counter = Counter()
    for _, _, attrs in graph.edges(data=True):
        edge_counter[attrs.get("edge_type", "unknown")] += 1

    return {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "node_types": dict(node_counter),
        "edge_types": dict(edge_counter),
    }
