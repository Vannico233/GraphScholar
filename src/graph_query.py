from collections import Counter
from typing import Any, Optional

import networkx as nx


def find_topic_node(graph: nx.Graph, topic: str) -> Optional[str]:
    normalized = topic.lower().strip()
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != "topic":
            continue
        name = str(attrs.get("name", "")).lower()
        if normalized == name or normalized in name or name in normalized:
            return node_id
    return None


def paper_node_to_dict(attrs: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": attrs.get("title", ""),
        "category": attrs.get("category", ""),
        "paper_type": attrs.get("paper_type", ""),
        "file_name": attrs.get("file_name", ""),
        "tasks": attrs.get("tasks", []),
        "applications": attrs.get("applications", []),
        "datasets": attrs.get("datasets", []),
    }


def query_methods_by_topic(graph: nx.Graph, topic: str, top_k: int = 10) -> list[str]:
    topic_node = find_topic_node(graph, topic)
    if not topic_node:
        return []

    counter = Counter()
    for paper_node in graph.neighbors(topic_node):
        if graph.nodes[paper_node].get("node_type") != "paper":
            continue
        for neighbor in graph.neighbors(paper_node):
            if graph.nodes[neighbor].get("node_type") == "method":
                counter[graph.nodes[neighbor].get("name", "")] += 1

    return [name for name, _ in counter.most_common(top_k)]


def query_papers_by_topic(graph: nx.Graph, topic: str, top_k: int = 10) -> list[dict[str, Any]]:
    topic_node = find_node_by_type(graph, "topic", topic)
    if not topic_node:
        return []

    papers = []
    for paper_node in graph.neighbors(topic_node):
        attrs = graph.nodes[paper_node]
        if attrs.get("node_type") != "paper":
            continue
        papers.append(paper_node_to_dict(attrs))

    papers.sort(key=lambda item: item["title"])
    return papers[:top_k]


def find_node_by_type(graph: nx.Graph, node_type: str, query: str) -> Optional[str]:
    normalized = query.lower().strip()
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != node_type:
            continue
        name = str(attrs.get("name", "")).lower()
        if normalized == name or normalized in name or name in normalized:
            return node_id
    return None


def extract_known_nodes(graph: nx.Graph, question: str, node_type: str) -> list[str]:
    question_lower = question.lower()
    question_tokens = {token for token in question_lower.replace("-", " ").split() if len(token) >= 4}
    scored_hits = []

    for _, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != node_type:
            continue
        name = str(attrs.get("name", ""))
        if not name:
            continue
        name_lower = name.lower()
        name_tokens = {token for token in name_lower.replace("-", " ").split() if len(token) >= 4}

        score = 0
        if name_lower in question_lower:
            score += 100
        overlap = len(question_tokens & name_tokens)
        score += overlap * 10
        if score > 0:
            scored_hits.append((score, name))

    scored_hits.sort(key=lambda item: (-item[0], -len(item[1])))
    return [name for _, name in scored_hits[:5]]


def query_bridge_papers(graph: nx.Graph, topic_a: str, topic_b: str) -> list[dict[str, Any]]:
    node_a = find_node_by_type(graph, "topic", topic_a)
    node_b = find_node_by_type(graph, "topic", topic_b)
    if not node_a or not node_b:
        return []

    papers_a = {node for node in graph.neighbors(node_a) if graph.nodes[node].get("node_type") == "paper"}
    papers_b = {node for node in graph.neighbors(node_b) if graph.nodes[node].get("node_type") == "paper"}
    shared = papers_a & papers_b

    results = []
    for node in sorted(shared):
        attrs = graph.nodes[node]
        results.append(paper_node_to_dict(attrs))
    return results


def query_neighbor_topics(graph: nx.Graph, topic: str, top_k: int = 10) -> list[str]:
    topic_node = find_topic_node(graph, topic)
    if not topic_node:
        return []

    counter = Counter()
    for paper_node in graph.neighbors(topic_node):
        if graph.nodes[paper_node].get("node_type") != "paper":
            continue
        for neighbor in graph.neighbors(paper_node):
            attrs = graph.nodes[neighbor]
            if attrs.get("node_type") == "topic" and neighbor != topic_node:
                counter[attrs.get("name", "")] += 1

    return [name for name, _ in counter.most_common(top_k)]


def extract_known_topics(graph: nx.Graph, question: str) -> list[str]:
    return extract_known_nodes(graph, question, "topic")


def extract_known_tasks(graph: nx.Graph, question: str) -> list[str]:
    return extract_known_nodes(graph, question, "task")


def extract_known_applications(graph: nx.Graph, question: str) -> list[str]:
    return extract_known_nodes(graph, question, "application")


def extract_known_datasets(graph: nx.Graph, question: str) -> list[str]:
    return extract_known_nodes(graph, question, "dataset")


def query_papers_by_node_type(graph: nx.Graph, node_type: str, value: str, top_k: int = 10) -> list[dict[str, Any]]:
    node_id = find_node_by_type(graph, node_type, value)
    if not node_id:
        return []

    papers = []
    for paper_node in graph.neighbors(node_id):
        attrs = graph.nodes[paper_node]
        if attrs.get("node_type") != "paper":
            continue
        papers.append(paper_node_to_dict(attrs))

    papers.sort(key=lambda item: item["title"])
    return papers[:top_k]


def route_graph_query(graph: nx.Graph, question: str) -> dict[str, Any]:
    hits = extract_known_topics(graph, question)
    task_hits = extract_known_tasks(graph, question)
    application_hits = extract_known_applications(graph, question)
    dataset_hits = extract_known_datasets(graph, question)
    lower = question.lower()

    if len(hits) >= 2 and ("连接" in question or "bridge" in lower or "between" in lower):
        papers = query_bridge_papers(graph, hits[0], hits[1])
        return {
            "query_type": "bridge_papers",
            "papers": papers,
            "summary_lines": [f"连接主题 {hits[0]} 和 {hits[1]} 的论文共有 {len(papers)} 篇。"],
            "graph_observations": [f"{hits[0]} 与 {hits[1]} 在当前图中存在论文级共现。"] if papers else [],
        }

    if task_hits and ("任务" in question or "下游" in question or "downstream" in lower):
        papers = query_papers_by_node_type(graph, "task", task_hits[0], top_k=5)
        return {
            "query_type": "papers_by_task",
            "papers": papers,
            "summary_lines": [f"任务 {task_hits[0]} 相关论文共有 {len(papers)} 篇。"],
            "graph_observations": [f"任务节点 {task_hits[0]} 连接了这些论文。"] if papers else [],
        }

    if application_hits and ("应用" in question or "场景" in question or "application" in lower):
        papers = query_papers_by_node_type(graph, "application", application_hits[0], top_k=5)
        return {
            "query_type": "papers_by_application",
            "papers": papers,
            "summary_lines": [f"应用场景 {application_hits[0]} 相关论文共有 {len(papers)} 篇。"],
            "graph_observations": [f"应用节点 {application_hits[0]} 连接了这些论文。"] if papers else [],
        }

    if dataset_hits and ("数据集" in question or "dataset" in lower or "benchmark" in lower):
        papers = query_papers_by_node_type(graph, "dataset", dataset_hits[0], top_k=5)
        return {
            "query_type": "papers_by_dataset",
            "papers": papers,
            "summary_lines": [f"数据集 {dataset_hits[0]} 相关论文共有 {len(papers)} 篇。"],
            "graph_observations": [f"数据集节点 {dataset_hits[0]} 连接了这些论文。"] if papers else [],
        }

    if hits and ("常见方法" in question or "代表方法" in question or "method" in lower or "methods" in lower):
        methods = query_methods_by_topic(graph, hits[0])
        papers = query_papers_by_topic(graph, hits[0], top_k=3)
        return {
            "query_type": "methods_by_topic",
            "papers": papers,
            "summary_lines": [f"主题 {hits[0]} 下常见方法: {', '.join(methods) if methods else '暂无明确方法节点'}"],
            "graph_observations": [f"{hits[0]} 通过 paper-method 边连接到 {len(methods)} 个方法节点。"] if methods else [],
        }

    if hits and ("邻近" in question or "相邻" in question or "neighbor" in lower):
        neighbors = query_neighbor_topics(graph, hits[0])
        papers = query_papers_by_topic(graph, hits[0], top_k=3)
        return {
            "query_type": "neighbor_topics",
            "papers": papers,
            "summary_lines": [f"主题 {hits[0]} 的邻近主题: {', '.join(neighbors) if neighbors else '暂无明显邻近主题'}"],
            "graph_observations": [f"{hits[0]} 与这些主题通过共享论文形成连接。"] if neighbors else [],
        }

    if hits:
        papers = query_papers_by_topic(graph, hits[0], top_k=5)
        methods = query_methods_by_topic(graph, hits[0], top_k=5)
        return {
            "query_type": "papers_by_topic",
            "papers": papers,
            "summary_lines": [f"主题 {hits[0]} 下代表论文 {len(papers)} 篇。"],
            "graph_observations": [f"相关方法包括: {', '.join(methods)}"] if methods else [],
        }

    return {
        "query_type": "unknown",
        "papers": [],
        "summary_lines": ["未能从问题中定位到已有主题节点。"],
        "graph_observations": [],
    }


def summarize_graph_relations(papers: list[dict[str, Any]]) -> list[str]:
    if len(papers) < 2:
        return []

    lines = []
    for i in range(min(len(papers), 3)):
        for j in range(i + 1, min(len(papers), 3)):
            a = papers[i]
            b = papers[j]
            relation_bits = []
            type_a = str(a.get("paper_type", "")).lower()
            type_b = str(b.get("paper_type", "")).lower()
            shared_tasks = sorted(set(a.get("tasks", [])) & set(b.get("tasks", [])))
            shared_datasets = sorted(set(a.get("datasets", [])) & set(b.get("datasets", [])))
            shared_apps = sorted(set(a.get("applications", [])) & set(b.get("applications", [])))

            if "survey" in {type_a, type_b}:
                relation_bits.append("综述关系")
            if {"benchmark", "evaluation"} & {type_a, type_b}:
                relation_bits.append("评测关系")
            if "application" in {type_a, type_b}:
                relation_bits.append("应用迁移")
            if shared_tasks:
                relation_bits.append(f"共享任务: {', '.join(shared_tasks[:2])}")
            if shared_datasets:
                relation_bits.append(f"共享数据集: {', '.join(shared_datasets[:2])}")
            if shared_apps:
                relation_bits.append(f"共享应用: {', '.join(shared_apps[:2])}")

            if relation_bits:
                lines.append(f'{a["title"]} <-> {b["title"]}: {"; ".join(relation_bits)}')
    return lines
