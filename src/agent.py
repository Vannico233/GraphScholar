import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from llm_api import build_llm_client, call_llm
from src.tools import (
    chunk_search,
    format_graph_query_result,
    infer_answer_plan,
    paper_retrieve,
    paper_search,
    paper_structurize,
    rewrite_query,
    topic_graph_query,
)
from src.self_check import self_check_answer


AGENT_SYSTEM_PROMPT = """
你是一个面向图学习文献分析的单智能体研究助手。

你会接收三类证据：
1. 文本检索证据
2. 全文 chunk 级证据
3. 图查询证据

请基于这些证据输出结构化、研究导向、可解释的回答。
回答必须使用以下标题：
代表工作
核心方法
优点
局限
启发
"""

FOCUS_TERMS = {
    "benchmark": ["benchmark", "evaluation"],
    "graphrag": ["graphrag", "graph rag"],
    "graph rag": ["graphrag", "graph rag"],
    "knowledge graph": ["knowledge graph"],
    "知识图谱": ["knowledge graph"],
    "图对比学习": ["contrastive"],
    "对比学习": ["contrastive"],
    "contrastive": ["contrastive"],
    "图预训练": ["pretraining", "pre-train", "graph pretraining"],
    "预训练": ["pretraining", "pre-train", "graph pretraining"],
    "pretraining": ["pretraining", "pre-train", "graph pretraining"],
    "图提示学习": ["prompt", "prompting", "graph prompting"],
    "提示学习": ["prompt", "prompting", "graph prompting"],
    "prompting": ["prompt", "prompting", "graph prompting"],
    "图表示学习": ["representation learning", "graph representation"],
    "图表征学习": ["representation learning", "graph representation"],
    "representation learning": ["representation learning", "graph representation"],
}


@dataclass
class AgentResult:
    question: str
    rewritten_query: str
    route_type: str
    search_results: List[Dict[str, Any]]
    retrieved_papers: List[Dict[str, Any]]
    evidence_chunks: List[Dict[str, Any]]
    graph_result: Optional[Dict[str, Any]]
    final_answer: str


def infer_focus_terms(question: str) -> List[str]:
    lowered = question.lower()
    terms: List[str] = []
    for marker, mapped_terms in FOCUS_TERMS.items():
        if marker.lower() in lowered:
            terms.extend(mapped_terms)
    return list(dict.fromkeys(terms))


def item_matches_focus(item: Dict[str, Any], focus_terms: List[str]) -> bool:
    if not focus_terms:
        return True

    searchable = " ".join(
        [
            str(item.get("title", "")),
            str(item.get("category", "")),
            " ".join(str(tag) for tag in item.get("tags", [])),
            str(item.get("abstract", "")),
            str(item.get("text", "")),
        ]
    ).lower()
    return any(term in searchable for term in focus_terms)


def filter_by_focus(items: List[Dict[str, Any]], focus_terms: List[str]) -> List[Dict[str, Any]]:
    if not focus_terms:
        return items
    filtered = [item for item in items if item_matches_focus(item, focus_terms)]
    return filtered or items


def classify_question_type(question: str) -> str:
    graph_keywords = ["常见方法", "代表方法", "连接", "关联", "两个主题", "邻近主题", "邻近", "topic", "bridge"]
    retrieval_keywords = ["总结", "代表工作", "综述", "优缺点", "介绍", "论文有哪些", "经典论文"]

    has_graph = any(keyword in question for keyword in graph_keywords) or any(
        keyword in question.lower() for keyword in ["method", "methods", "neighbor", "bridge"]
    )
    has_retrieval = any(keyword in question for keyword in retrieval_keywords)

    if has_graph and has_retrieval:
        return "hybrid"
    if has_graph:
        return "graph"
    return "retrieval"


class ResearchAssistantAgent:
    def __init__(self, use_llm: bool = True):
        self.requested_llm = use_llm
        self.use_llm = False
        self.client = None
        if use_llm:
            try:
                self.client = build_llm_client()
                self.use_llm = True
            except Exception:
                self.client = None
                self.use_llm = False

    def run(self, question: str, top_k: int = 5) -> AgentResult:
        rewritten_query = rewrite_query(question)
        route_type = classify_question_type(question)
        answer_plan = infer_answer_plan(question)

        search_results: List[Dict[str, Any]] = []
        retrieved_papers: List[Dict[str, Any]] = []
        evidence_chunks: List[Dict[str, Any]] = []
        graph_result: Optional[Dict[str, Any]] = None

        if route_type in {"retrieval", "hybrid"}:
            focus_terms = infer_focus_terms(rewritten_query)
            search_results = filter_by_focus(paper_search(rewritten_query, top_k=max(top_k * 2, 10)), focus_terms)[
                :top_k
            ]
            evidence_chunks = filter_by_focus(chunk_search(rewritten_query, top_k=12), focus_terms)[:6]

            file_names = [item["file_name"] for item in search_results[:3]]
            if evidence_chunks:
                file_names.extend(chunk["file_name"] for chunk in evidence_chunks[:3])
            retrieved_papers = paper_retrieve(list(dict.fromkeys(file_names)))

        if route_type in {"graph", "hybrid"}:
            graph_result = topic_graph_query(rewritten_query)

        if self.use_llm:
            draft_answer = self._build_llm_answer(
                question, rewritten_query, route_type, answer_plan, retrieved_papers, evidence_chunks, graph_result
            )
            self_check_payload = {
                "retrieved_papers": retrieved_papers,
                "evidence_chunks": evidence_chunks,
            }
            checked = self_check_answer(
                question=question,
                rewritten_query=rewritten_query,
                retrieved_evidence=self_check_payload,
                draft_answer=draft_answer,
                use_llm=self.use_llm,
                client=self.client,
            )
            final_answer = checked.get("improved_answer", draft_answer)
        else:
            final_answer = self._build_local_answer(
                question, rewritten_query, route_type, answer_plan, retrieved_papers, evidence_chunks, graph_result
            )

        return AgentResult(
            question=question,
            rewritten_query=rewritten_query,
            route_type=route_type,
            search_results=search_results,
            retrieved_papers=retrieved_papers,
            evidence_chunks=evidence_chunks,
            graph_result=graph_result,
            final_answer=final_answer,
        )

    def _build_local_answer(
        self,
        question: str,
        rewritten_query: str,
        route_type: str,
        answer_plan: str,
        retrieved_papers: List[Dict[str, Any]],
        evidence_chunks: List[Dict[str, Any]],
        graph_result: Optional[Dict[str, Any]],
    ) -> str:
        if route_type == "retrieval":
            return paper_structurize(question, retrieved_papers, evidence_chunks, answer_plan=answer_plan)
        if route_type == "graph":
            return format_graph_query_result(question, graph_result or {}, answer_plan=answer_plan)

        retrieval_text = paper_structurize(question, retrieved_papers, evidence_chunks, answer_plan=answer_plan)
        graph_text = format_graph_query_result(question, graph_result or {}, answer_plan=answer_plan)
        return f"{graph_text}\n\n五、文本检索补充\n{retrieval_text}"

    def _build_llm_answer(
        self,
        question: str,
        rewritten_query: str,
        route_type: str,
        answer_plan: str,
        retrieved_papers: List[Dict[str, Any]],
        evidence_chunks: List[Dict[str, Any]],
        graph_result: Optional[Dict[str, Any]],
    ) -> str:
        evidence = {
            "original_question": question,
            "rewritten_query": rewritten_query,
            "route_type": route_type,
            "answer_plan": answer_plan,
            "retrieved_papers": [
                {
                    "title": paper["title"],
                    "category": paper["category"],
                    "tags": paper["tags"],
                    "paper_type": paper.get("paper_type", ""),
                    "tasks": paper.get("tasks", []),
                    "applications": paper.get("applications", []),
                    "datasets": paper.get("datasets", []),
                    "method_summary": paper.get("method_summary", "")[:800],
                    "contribution_summary": paper.get("contribution_summary", "")[:800],
                    "abstract": paper["abstract"][:1500],
                }
                for paper in retrieved_papers
            ],
            "evidence_chunks": [
                {
                    "title": chunk["title"],
                    "page": chunk["page"],
                    "text": chunk["text"][:600],
                }
                for chunk in evidence_chunks[:5]
            ],
            "graph_result": graph_result,
        }

        prompt = (
            f"用户原始问题:\n{question}\n\n"
            f"改写后问题:\n{rewritten_query}\n\n"
            f"回答规划:\n{answer_plan}\n\n"
            f"证据:\n{json.dumps(evidence, ensure_ascii=False, indent=2)}\n\n"
            "请按照回答规划组织内容。优先使用证据中的任务、数据集、应用场景、方法摘要和关系解释，"
            "不要只列论文名；如果是分类综述，请按方法论文、评测论文、综述论文、应用论文等类别组织。"
        )
        return call_llm(
            prompt=prompt,
            system_prompt=AGENT_SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=2500,
            client=self.client,
        )
