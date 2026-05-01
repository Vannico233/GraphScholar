import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.build_graph import build_paper_graph_from_json, get_graph_stats
from src.graph_query import route_graph_query, summarize_graph_relations


BASE_DIR = Path(__file__).resolve().parent.parent
PAPERS_PATH = BASE_DIR / "outputs" / "paper_summaries.json"
CHUNKS_PATH = BASE_DIR / "outputs" / "paper_chunks.json"
GRAPH_CACHE_PATH = BASE_DIR / "outputs" / "paper_graph.json"

QUERY_EXPANSIONS = {
    "谱方法": ["spectral methods", "spectral graph theory", "graph Fourier", "Laplacian", "eigenvector"],
    "高频": ["high-frequency", "high pass", "high-pass"],
    "低频": ["low-frequency", "low pass", "low-pass"],
    "高低频": ["high-frequency", "low-frequency", "spectral bias"],
    "异配图": ["heterophily", "heterophilous graph"],
    "异构图": ["heterogeneous graph"],
    "链接预测": ["link prediction"],
    "节点分类": ["node classification"],
    "图分类": ["graph classification"],
    "图问答": ["question answering"],
    "问答": ["question answering"],
    "数据集": ["dataset", "benchmark"],
    "应用场景": ["application"],
    "推荐": ["recommendation"],
    "图对比学习": ["Graph Contrastive Learning", "contrastive learning", "representative papers"],
    "对比学习": ["contrastive learning", "representative papers"],
    "图预训练": ["Graph Pretraining", "pretraining", "representative papers"],
    "预训练": ["pretraining", "representative papers"],
    "图提示学习": ["Graph Prompting", "prompting", "representative methods"],
    "提示学习": ["prompting", "representative methods"],
    "图表示学习": ["Graph Representation Learning", "representation learning"],
    "图表征学习": ["Graph Representation Learning", "representation learning"],
    "知识图谱": ["Knowledge Graph"],
    "GraphRAG 和 KG": ["GraphRAG", "Knowledge Graph", "technical connections", "representative papers"],
    "graph prompt": ["Graph Prompting", "representative methods", "advantages", "limitations"],
    "graph prompting": ["Graph Prompting", "representative methods", "advantages", "limitations"],
    "graphrag and kg": ["GraphRAG", "Knowledge Graph", "technical connections", "representative papers"],
}

TASK_QUERY_ALIASES = {
    "link prediction": ["link prediction", "链接预测"],
    "node classification": ["node classification", "节点分类"],
    "graph classification": ["graph classification", "图分类"],
    "question answering": ["question answering", "问答", "图问答"],
    "summarization": ["summarization", "摘要", "总结"],
    "retrieval": ["retrieval", "检索"],
    "recommendation": ["recommendation", "推荐"],
    "reasoning": ["reasoning", "推理"],
    "clustering": ["clustering", "聚类"],
}

ANSWER_PLAN_HINTS = {
    "reading_route": ["阅读路线", "先读", "入门", "学习路径", "怎么读"],
    "mechanism_explanation": ["机制", "原理", "谱方法", "高频", "低频", "异配", "heterophily", "架构设计", "设计"],
    "experimental_comparison": ["比较", "对比", "性能", "效果", "表现", "benchmark", "实验"],
    "paper_recommendation": ["推荐", "找", "近期", "最近", "表现好", "优先看", "适合"],
    "classification_review": ["有哪些", "分类", "综述", "梳理", "代表论文", "代表工作", "survey", "review"],
    "background_summary": ["背景", "概述", "介绍", "入门", "overview"],
}

GENERIC_QUERY_TOKENS = {
    "graph",
    "graphs",
    "paper",
    "papers",
    "method",
    "methods",
    "representative",
    "related",
    "focus",
    "retrieval",
    "key",
    "evidence",
    "advantages",
    "limitations",
}

STRONG_QUERY_TOKEN_WEIGHTS = {
    "graphrag": 4,
    "graph rag": 4,
    "benchmark": 5,
    "evaluation": 3,
    "knowledge graph": 5,
    "graph contrastive learning": 5,
    "contrastive": 4,
    "graph pretraining": 5,
    "pretraining": 4,
    "pre-train": 4,
    "graph prompting": 5,
    "prompting": 4,
    "graph representation learning": 5,
    "representation learning": 4,
}

STRUCTURED_FIELD_WEIGHTS = {
    "tasks": 7,
    "applications": 5,
    "datasets": 4,
    "paper_type": 3,
    "method_summary": 2,
    "contribution_summary": 2,
}

EVIDENCE_SECTION_BONUS = {
    "abstract": 20,
    "introduction": 12,
    "method": 18,
    "approach": 16,
    "model": 12,
    "experiment": 16,
    "experiments": 16,
    "results": 16,
    "evaluation": 16,
    "discussion": 10,
    "conclusion": 10,
}

EVIDENCE_SECTION_PENALTY = {
    "references": 120,
    "bibliography": 120,
    "acknowledgment": 60,
    "acknowledgements": 60,
    "appendix": 20,
}


def load_papers() -> List[Dict[str, Any]]:
    return json.loads(PAPERS_PATH.read_text(encoding="utf-8"))


def load_chunks() -> List[Dict[str, Any]]:
    if not CHUNKS_PATH.exists():
        return []
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))


def extract_requested_tasks(question: str) -> List[str]:
    lowered = question.lower()
    requested = []
    for task_name, aliases in TASK_QUERY_ALIASES.items():
        if any(alias.lower() in lowered for alias in aliases):
            requested.append(task_name)
    return list(dict.fromkeys(requested))


def infer_answer_plan(question: str) -> str:
    lowered = question.lower()
    for plan, hints in ANSWER_PLAN_HINTS.items():
        if any(hint.lower() in lowered for hint in hints):
            return plan
    if "哪些论文" in question or "有哪些" in question:
        return "classification_review"
    return "background_summary"


def rewrite_query(question: str) -> str:
    for pattern, expansions in QUERY_EXPANSIONS.items():
        if pattern.lower() in question.lower():
            return f"{question.strip()} | Retrieval focus: {', '.join(expansions)}"

    if "关系" in question or "关联" in question:
        return f"{question.strip()} | Focus on representative papers, technical connections, and limitations."
    if "怎么" in question or "怎么样" in question:
        return f"{question.strip()} | Focus on representative methods, strengths, limitations, and research insights."
    if "有哪些" in question:
        return f"{question.strip()} | Focus on representative papers, methods, and key evidence."
    return question.strip()


def tokenize(text: str) -> List[str]:
    lowered = text.lower()
    expanded_tokens: List[str] = []

    for phrase, mapped_tokens in QUERY_EXPANSIONS.items():
        if phrase.lower() in lowered:
            expanded_tokens.extend(token.lower() for token in mapped_tokens)
            lowered = lowered.replace(phrase.lower(), " ")

    expanded_tokens.extend(re.findall(r"[a-zA-Z0-9\-]+", lowered))
    return expanded_tokens


def score_evidence_text(text: str) -> int:
    lowered = text.lower()
    score = 0
    for section, bonus in EVIDENCE_SECTION_BONUS.items():
        if section in lowered:
            score += bonus
    for section, penalty in EVIDENCE_SECTION_PENALTY.items():
        if section in lowered:
            score -= penalty
    if re.search(r"\breferences\b|\bbibliography\b", lowered):
        score -= 100
    if re.search(r"\bdoi\b|\barxiv\b", lowered) and len(lowered) < 500:
        score -= 15
    return score


def rank_evidence_chunks(question: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = []
    question_tokens = {token for token in tokenize(question) if token not in GENERIC_QUERY_TOKENS}
    for chunk in chunks:
        text = str(chunk.get("text", ""))
        base = int(chunk.get("score", 0))
        bonus = score_evidence_text(text)
        overlap_bonus = 0
        lowered = text.lower()
        for token in question_tokens:
            if token in lowered:
                overlap_bonus += 2
        ranked.append((base + bonus + overlap_bonus, chunk))
    ranked.sort(key=lambda item: (-item[0], item[1].get("page", 0)))
    return [chunk for _, chunk in ranked]


def compute_relevance(question: str, paper: Dict[str, Any]) -> tuple[int, List[str]]:
    question_tokens = {token for token in tokenize(question) if token not in GENERIC_QUERY_TOKENS}
    if not question_tokens:
        return 0, []

    title = str(paper.get("title", "")).lower()
    abstract = str(paper.get("abstract", "")).lower()
    category = str(paper.get("category", "")).lower()
    tags = " ".join(str(tag) for tag in paper.get("tags", [])).lower()
    tasks = " ".join(str(item) for item in paper.get("tasks", [])).lower()
    applications = " ".join(str(item) for item in paper.get("applications", [])).lower()
    datasets = " ".join(str(item) for item in paper.get("datasets", [])).lower()
    paper_type = str(paper.get("paper_type", "")).lower()
    method_summary = str(paper.get("method_summary", "")).lower()
    contribution_summary = str(paper.get("contribution_summary", "")).lower()

    score = 0
    evidence = []
    for token in question_tokens:
        token_weight = STRONG_QUERY_TOKEN_WEIGHTS.get(token, 1)
        if token in title:
            score += 5 * token_weight
            evidence.append(f"title:{token}")
        if token in category:
            score += 4 * token_weight
            evidence.append(f"category:{token}")
        if token in tags:
            score += 3 * token_weight
            evidence.append(f"tag:{token}")
        if token in abstract:
            score += token_weight
            evidence.append(f"abstract:{token}")

    structured_fields = (
        (tasks, "tasks", STRUCTURED_FIELD_WEIGHTS["tasks"]),
        (applications, "applications", STRUCTURED_FIELD_WEIGHTS["applications"]),
        (datasets, "datasets", STRUCTURED_FIELD_WEIGHTS["datasets"]),
        (paper_type, "paper_type", STRUCTURED_FIELD_WEIGHTS["paper_type"]),
        (method_summary, "method_summary", STRUCTURED_FIELD_WEIGHTS["method_summary"]),
        (contribution_summary, "contribution_summary", STRUCTURED_FIELD_WEIGHTS["contribution_summary"]),
    )
    for field_text, field_name, field_weight in structured_fields:
        for token in question_tokens:
            if token in field_text:
                score += field_weight * max(1, STRONG_QUERY_TOKEN_WEIGHTS.get(token, 1))
                evidence.append(f"{field_name}:{token}")
    return score, evidence


def task_overlap(requested_tasks: List[str], paper_tasks: List[str]) -> bool:
    requested = {task.lower() for task in requested_tasks}
    paper_set = {task.lower() for task in paper_tasks}
    return bool(requested & paper_set)


def adjust_for_requested_tasks(
    ranked: List[Dict[str, Any]],
    requested_tasks: List[str],
    top_k: int,
) -> List[Dict[str, Any]]:
    if not requested_tasks:
        return ranked[:top_k]

    matched = []
    unmatched = []
    for item in ranked:
        paper_tasks = [str(task).lower() for task in item.get("tasks", [])]
        if task_overlap(requested_tasks, paper_tasks):
            item = dict(item)
            item["score"] = item["score"] + 40
            item.setdefault("match_evidence", []).append("tasks:requested")
            matched.append(item)
        else:
            item = dict(item)
            if paper_tasks:
                item["score"] = max(1, item["score"] - 12)
            unmatched.append(item)

    matched.sort(key=lambda item: (-item["score"], item["title"]))
    unmatched.sort(key=lambda item: (-item["score"], item["title"]))
    return (matched + unmatched)[:top_k]


def compute_chunk_relevance(question: str, chunk: Dict[str, Any]) -> tuple[int, List[str]]:
    question_tokens = {token for token in tokenize(question) if token not in GENERIC_QUERY_TOKENS}
    if not question_tokens:
        return 0, []

    title = str(chunk.get("title", "")).lower()
    text = str(chunk.get("text", "")).lower()
    score = 0
    evidence = []
    for token in question_tokens:
        token_weight = STRONG_QUERY_TOKEN_WEIGHTS.get(token, 1)
        if token in title:
            score += 3 * token_weight
            evidence.append(f"title:{token}")
        if token in text:
            score += 2 * token_weight
            evidence.append(f"text:{token}")
    return score, evidence


def paper_search(question: str, top_k: int = 5) -> List[Dict[str, Any]]:
    ranked = []
    requested_tasks = extract_requested_tasks(question)
    for paper in load_papers():
        score, evidence = compute_relevance(question, paper)
        if score > 0:
            ranked.append(
                {
                    "file_name": paper["file_name"],
                    "title": paper["title"],
                    "category": paper["category"],
                    "paper_type": paper.get("paper_type", ""),
                    "tags": paper["tags"],
                    "tasks": paper.get("tasks", []),
                    "applications": paper.get("applications", []),
                    "datasets": paper.get("datasets", []),
                    "method_summary": paper.get("method_summary", ""),
                    "contribution_summary": paper.get("contribution_summary", ""),
                    "abstract": paper["abstract"],
                    "score": score,
                    "match_evidence": evidence,
                }
            )
    ranked.sort(key=lambda item: (-item["score"], item["title"]))
    return adjust_for_requested_tasks(ranked, requested_tasks, top_k)


def chunk_search(question: str, top_k: int = 8) -> List[Dict[str, Any]]:
    ranked = []
    for chunk in load_chunks():
        score, evidence = compute_chunk_relevance(question, chunk)
        if score > 0:
            ranked.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "file_name": chunk["file_name"],
                    "title": chunk["title"],
                    "page": chunk["page"],
                    "text": chunk["text"],
                    "score": score,
                    "match_evidence": evidence,
                }
            )
    ranked.sort(key=lambda item: (-item["score"], item["title"], item["page"]))
    return rank_evidence_chunks(question, ranked)[:top_k]


def paper_retrieve(file_names: List[str]) -> List[Dict[str, Any]]:
    wanted = set(file_names)
    return [paper for paper in load_papers() if paper["file_name"] in wanted]


def build_citation_lines(chunks: List[Dict[str, Any]], limit: int = 3) -> List[str]:
    lines = []
    for chunk in chunks[:limit]:
        quote = extract_evidence_snippet(str(chunk.get("text", "")))
        lines.append(f'- {chunk["title"]} [p.{chunk["page"]}]: "{quote}..."')
    return lines


def extract_evidence_snippet(text: str, limit: int = 220) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    sentences = re.split(r"(?<=[。.!?])\s+", cleaned)
    candidates = []
    for sentence in sentences:
        sent_lower = sentence.lower()
        if any(term in sent_lower for term in ("references", "bibliography", "acknowledgment", "acknowledgements")):
            continue
        score = score_evidence_text(sentence)
        if any(term in sent_lower for term in ("method", "experiment", "results", "evaluation", "conclusion", "approach")):
            score += 8
        if any(term in sent_lower for term in ("abstract", "introduction")):
            score += 4
        if sentence.strip():
            candidates.append((score, sentence.strip()))
    if candidates:
        candidates.sort(key=lambda item: (-item[0], len(item[1])))
        return candidates[0][1][:limit]
    return cleaned[:limit]


def summarize_paper_relations(papers: List[Dict[str, Any]], limit: int = 3) -> List[str]:
    if len(papers) < 2:
        return []

    relations = []
    for i in range(min(len(papers), limit)):
        for j in range(i + 1, min(len(papers), limit)):
            a = papers[i]
            b = papers[j]
            shared_tasks = sorted(set(a.get("tasks", [])) & set(b.get("tasks", [])))
            shared_datasets = sorted(set(a.get("datasets", [])) & set(b.get("datasets", [])))
            shared_apps = sorted(set(a.get("applications", [])) & set(b.get("applications", [])))
            relation_types = []

            if a.get("paper_type") == "survey" or b.get("paper_type") == "survey":
                relation_types.append("综述关系")
            if a.get("paper_type") in {"benchmark", "evaluation"} or b.get("paper_type") in {"benchmark", "evaluation"}:
                relation_types.append("评测关系")
            if a.get("paper_type") == "application" or b.get("paper_type") == "application":
                relation_types.append("应用迁移")
            if shared_tasks:
                relation_types.append(f"共享任务: {', '.join(shared_tasks[:2])}")
            if shared_datasets:
                relation_types.append(f"共享数据集: {', '.join(shared_datasets[:2])}")
            if shared_apps:
                relation_types.append(f"共享应用: {', '.join(shared_apps[:2])}")

            if relation_types:
                relations.append(f'{a["title"]} <-> {b["title"]}: {"; ".join(relation_types)}')

    return relations[:limit]


def paper_structurize(
    question: str,
    papers: List[Dict[str, Any]],
    evidence_chunks: Optional[List[Dict[str, Any]]] = None,
    answer_plan: str = "background_summary",
) -> str:
    if not papers:
        return (
            f"问题: {question}\n\n"
            "代表工作\n未检索到足够相关的论文。\n\n"
            "核心方法\n请尝试换更具体的主题词、方法词或关系描述。\n\n"
            "优点\n当前系统支持全文 chunk 检索和页码级引用。\n\n"
            "局限\n当前检索仍以关键词匹配为主，语义泛化能力有限。\n\n"
            "启发\n可以先通过 query rewrite 补全主题词和方法词，再重新检索。"
        )

    chosen_papers = papers[:3]
    lines = [f"问题: {question}", "", "代表工作"]
    for paper in chosen_papers:
        paper_type = paper.get("paper_type") or paper.get("category", "unknown")
        lines.append(f'- {paper["title"]} ({paper_type}) | 标签: {", ".join(paper["tags"])}')

    if any(paper.get("tasks") or paper.get("applications") or paper.get("datasets") for paper in chosen_papers):
        lines.extend(["", "结构化信息"])
        for paper in chosen_papers:
            task_part = f"tasks={', '.join(paper.get('tasks', []))}" if paper.get("tasks") else ""
            app_part = f"applications={', '.join(paper.get('applications', []))}" if paper.get("applications") else ""
            dataset_part = f"datasets={', '.join(paper.get('datasets', []))}" if paper.get("datasets") else ""
            parts = [part for part in [task_part, app_part, dataset_part] if part]
            if parts:
                lines.append(f'- {paper["title"]}: ' + " | ".join(parts))

    plan_labels = {
        "classification_review": "分类结果",
        "paper_recommendation": "推荐理由",
        "mechanism_explanation": "机制主线",
        "experimental_comparison": "实验对比",
        "reading_route": "阅读路线",
        "background_summary": "背景梳理",
    }
    selected_heading = plan_labels.get(answer_plan, "核心方法")
    lines.extend(["", selected_heading])

    if answer_plan == "classification_review":
        groups = {"方法论文": [], "评测论文": [], "综述论文": [], "应用论文": [], "其他": []}
        for paper in chosen_papers:
            paper_type = str(paper.get("paper_type", "")).lower()
            if paper_type in {"method", "system", "theory", "other"}:
                groups["方法论文"].append(paper["title"])
            elif paper_type in {"benchmark", "evaluation"}:
                groups["评测论文"].append(paper["title"])
            elif paper_type == "survey":
                groups["综述论文"].append(paper["title"])
            elif paper_type == "application":
                groups["应用论文"].append(paper["title"])
            else:
                groups["其他"].append(paper["title"])
        for group_name, titles in groups.items():
            if titles:
                lines.append(f'- {group_name}: {", ".join(titles)}')
    elif answer_plan == "paper_recommendation":
        for paper in chosen_papers:
            reason_parts = []
            if paper.get("tasks"):
                reason_parts.append(f"任务: {', '.join(paper['tasks'][:2])}")
            if paper.get("datasets"):
                reason_parts.append(f"数据集: {', '.join(paper['datasets'][:2])}")
            if paper.get("method_summary"):
                reason_parts.append(f"方法: {paper['method_summary'][:120]}")
            lines.append(f'- {paper["title"]}: ' + " | ".join(reason_parts or ["适合作为代表工作"]))
    elif answer_plan == "mechanism_explanation":
        for paper in chosen_papers:
            lines.append(f'- {paper["title"]}: {paper.get("method_summary") or paper.get("contribution_summary") or paper["abstract"][:220]}...')
    elif answer_plan == "experimental_comparison":
        for paper in chosen_papers:
            parts = []
            if paper.get("tasks"):
                parts.append(f"任务={', '.join(paper['tasks'][:3])}")
            if paper.get("datasets"):
                parts.append(f"数据集={', '.join(paper['datasets'][:3])}")
            if paper.get("applications"):
                parts.append(f"应用={', '.join(paper['applications'][:2])}")
            lines.append(f'- {paper["title"]}: ' + " | ".join(parts or [paper["abstract"][:220]]))
    elif answer_plan == "reading_route":
        for idx, paper in enumerate(chosen_papers, start=1):
            lines.append(f"- Step {idx}: {paper['title']} ({paper.get('paper_type', paper.get('category', 'unknown'))})")
    else:
        if evidence_chunks:
            for chunk in rank_evidence_chunks(question, evidence_chunks)[:3]:
                snippet = extract_evidence_snippet(str(chunk["text"]), limit=260)
                lines.append(f'- {chunk["title"]} [p.{chunk["page"]}]: {snippet}...')
        else:
            for paper in chosen_papers:
                snippet = extract_evidence_snippet(str(paper["abstract"]), limit=260)
                lines.append(f'- {paper["title"]}: {snippet}...')

    relation_lines = summarize_paper_relations(chosen_papers)
    if relation_lines:
        lines.extend(["", "关系解释"])
        lines.extend(f"- {line}" for line in relation_lines)

    lines.extend(["", "优点"])
    lines.append("- 可以基于全文 chunk 而不是只基于摘要组织回答。")
    lines.append("- 可以附带页码级证据引用，便于回查原文。")

    lines.extend(["", "局限"])
    lines.append("- 当前局限分析在无 LLM 模式下仍偏模板化。")
    lines.append("- 当前检索主要依赖关键词重叠，不是语义向量检索。")

    lines.extend(["", "启发"])
    lines.append("- 命中的全文证据更适合支持研究型问答。")
    lines.append("- 后续适合继续加入 rerank、sentence extraction 和 stronger self-check。")

    if evidence_chunks:
        lines.extend(["", "证据引用"])
        lines.extend(build_citation_lines(rank_evidence_chunks(question, evidence_chunks)))

    return "\n".join(lines)


def get_or_build_graph():
    return build_paper_graph_from_json(PAPERS_PATH, save_path=GRAPH_CACHE_PATH)


def topic_graph_query(question: str) -> Dict[str, Any]:
    graph = get_or_build_graph()
    result = route_graph_query(graph, question)
    result["graph_stats"] = get_graph_stats(graph)
    return result


def format_graph_query_result(question: str, result: Dict[str, Any], answer_plan: str = "background_summary") -> str:
    lines = [f"问题: {question}", "", "一、图查询结果"]
    for line in result.get("summary_lines", []):
        lines.append(f"- {line}")

    lines.extend(["", "二、代表论文"])
    if result.get("papers"):
        for paper in result["papers"]:
            paper_type = paper.get("paper_type", "") or paper.get("category", "Unknown")
            lines.append(f'- {paper["title"]} ({paper_type})')
    else:
        lines.append("- 暂无直接命中的代表论文。")

    lines.extend(["", "三、图结构观察"])
    if result.get("graph_observations"):
        for item in result["graph_observations"]:
            lines.append(f"- {item}")
    else:
        lines.append("- 当前图上没有足够的结构性观察。")

    relation_lines = summarize_graph_relations(result.get("papers", []))
    if relation_lines:
        lines.extend(["", "四、关系解释"])
        lines.extend(f"- {line}" for line in relation_lines)

    stats = result.get("graph_stats", {})
    lines.extend(["", "五、启发"])
    lines.append(
        f'- 当前图包含 {stats.get("num_nodes", 0)} 个节点、{stats.get("num_edges", 0)} 条边，可用于主题-方法-论文共现分析。'
    )
    lines.append("- 图查询适合回答主题关系、桥接论文、近邻主题和常见方法问题。")
    if answer_plan == "paper_recommendation":
        lines.append("- 对推荐类问题，建议优先看带有任务、数据集和 paper_type 的论文。")
    return "\n".join(lines)
