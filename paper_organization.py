import json
import re
from pathlib import Path
from typing import Any, Optional, TypedDict

try:
    import fitz
except ImportError:  # pragma: no cover
    fitz = None
from langgraph.graph import END, START, StateGraph
from llm_api import call_llm


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
SUMMARY_PATH = OUTPUT_DIR / "paper_summaries.json"
CHUNKS_PATH = OUTPUT_DIR / "paper_chunks.json"
METADATA_CACHE_PATH = OUTPUT_DIR / "paper_metadata_cache.json"


class PaperState(TypedDict):
    pdf_path: str
    title: str
    raw_text: str
    abstract: str
    tags: list[str]
    category: str
    chunks: list[dict[str, Any]]


EXTRACTION_SYSTEM_PROMPT = """
You extract structured paper metadata for a research assistant.

Only use the provided title, abstract, and selected passages. Do not invent details.
Return valid JSON only with these keys:
- paper_type: one of "method", "survey", "benchmark", "application", "system", "theory", "dataset", "other"
- tasks: list of downstream tasks or research tasks explicitly targeted by the paper
- applications: list of concrete application domains or scenarios, only when the paper is application-oriented
- datasets: list of dataset or benchmark names explicitly mentioned in the evidence
- method_summary: one short sentence describing the main method or approach
- contribution_summary: one short sentence describing the main contribution
- confidence: one of "high", "medium", "low"

Rules:
- Prefer tasks from experiments, evaluation, or problem setting sections.
- If the paper is a survey or benchmark, keep tasks empty unless specific downstream tasks are explicitly evaluated.
- If the paper is application-oriented, applications should be specific domains or use cases, not generic labels.
- If the evidence is weak, use empty lists instead of guessing.
"""


KEYWORD_RULES = {
    "graphrag": "GraphRAG",
    "graph rag": "GraphRAG",
    "graph neural network": "GNN",
    "graph convolutional network": "GCN",
    "contrastive": "Graph Contrastive Learning",
    "self-supervised": "Graph Self-Supervised Learning",
    "heterogeneous graph": "Heterogeneous Graph",
    "knowledge graph": "Knowledge Graph",
    "representation learning": "Graph Representation Learning",
    "pre-train": "Graph Pretraining",
    "pretraining": "Graph Pretraining",
    "prompt": "Graph Prompting Learning",
}

RAG_KEYWORDS = (
    "retrieval-augmented generation",
    "retrieval augmented generation",
    "graphrag",
    "graph rag",
)
GRAPH_REPRESENTATION_KEYWORDS = (
    "graph neural network",
    "graph convolutional network",
)
EXTRACTION_HINT_KEYWORDS = (
    "task",
    "tasks",
    "downstream",
    "application",
    "applications",
    "dataset",
    "datasets",
    "benchmark",
    "evaluation",
    "experiment",
    "experiments",
    "result",
    "results",
    "use case",
    "use cases",
    "problem setting",
    "setting",
)
ABSTRACT_PATTERN = re.compile(
    r"abstract\s*[:\-]?\s*(.*?)\s*(?:1\s+introduction|i\.\s*introduction|introduction)",
    re.IGNORECASE | re.DOTALL,
)
JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def contains_any_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def clean_title_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.:;!?])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]])", r"\1", text)
    text = text.strip("-_.,:;")
    return merge_split_title_tokens(text)


def merge_split_title_tokens(text: str) -> str:
    tokens = text.split()
    merged = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if len(token) == 1 and token.isalpha():
            letters = [token]
            j = i + 1
            while j < len(tokens) and len(tokens[j]) == 1 and tokens[j].isalpha():
                letters.append(tokens[j])
                j += 1
            if len(letters) >= 3:
                merged.append("".join(letters))
                i = j
                continue
        merged.append(token)
        i += 1
    return " ".join(merged)


def append_span_text(parts: list[str], span_text: str, prev_bbox, curr_bbox) -> None:
    if not span_text:
        return
    if not parts:
        parts.append(span_text)
        return

    previous_text = parts[-1]
    gap = float(curr_bbox[0]) - float(prev_bbox[2]) if prev_bbox and curr_bbox else None
    should_merge = False

    if gap is not None and gap <= 1.5:
        should_merge = True
    elif previous_text and previous_text[-1].isalpha() and span_text[0].islower():
        should_merge = True

    if should_merge:
        parts[-1] = previous_text + span_text
    else:
        parts.append(span_text)


def looks_like_title(text: str) -> bool:
    lowered = text.lower().strip()
    if not lowered:
        return False
    if lowered in {"abstract", "introduction", "contents"}:
        return False
    if "@" in lowered:
        return False
    if lowered.startswith("arxiv"):
        return False
    if len(lowered) < 8:
        return False
    return True


def extract_title_from_first_page(doc: fitz.Document, fallback_title: str) -> str:
    if len(doc) == 0:
        return fallback_title

    page = doc.load_page(0)
    page_dict = page.get_text("dict")
    candidates = []

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        block_text_parts = []
        max_size = 0.0
        for line in block.get("lines", []):
            previous_bbox = None
            for span in line.get("spans", []):
                span_text = clean_title_text(span.get("text", ""))
                if not span_text:
                    continue
                append_span_text(block_text_parts, span_text, previous_bbox, span.get("bbox"))
                max_size = max(max_size, float(span.get("size", 0.0)))
                previous_bbox = span.get("bbox")
        block_text = clean_title_text(" ".join(block_text_parts))
        if looks_like_title(block_text):
            candidates.append((max_size, block_text))

    if candidates:
        candidates.sort(key=lambda item: (-item[0], len(item[1])))
        return candidates[0][1]

    for line in page.get_text("text").splitlines():
        candidate = clean_title_text(line)
        if looks_like_title(candidate):
            return candidate
    return fallback_title


def infer_category(text: str) -> str:
    normalized = text.lower()
    if contains_any_keyword(normalized, RAG_KEYWORDS):
        return "GraphRAG"
    if "contrastive" in normalized:
        return "Graph Contrastive Learning"
    if "pre-train" in normalized or "pretraining" in normalized:
        return "Graph Pretraining"
    if contains_any_keyword(normalized, GRAPH_REPRESENTATION_KEYWORDS):
        return "Graph Representation Learning"
    return "Graph Representation Learning"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json_file(path: Path, data: Any) -> None:
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def parse_json_response(raw: str) -> Optional[dict[str, Any]]:
    candidate = strip_code_fences(raw)
    match = JSON_BLOCK_PATTERN.search(candidate)
    if match:
        candidate = match.group(0)
    try:
        parsed = json.loads(candidate)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def normalize_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    seen = set()
    for item in value:
        if isinstance(item, str):
            text = normalize_text(item)
        elif isinstance(item, dict):
            text = normalize_text(str(item.get("name", "")))
        else:
            text = normalize_text(str(item))
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(text)
    return result


def select_extraction_passages(chunks: list[dict[str, Any]], limit: int = 8) -> list[str]:
    scored = []
    for index, chunk in enumerate(chunks):
        text = normalize_text(str(chunk.get("text", "")))
        if not text:
            continue
        lowered = text.lower()
        score = 0
        for keyword in EXTRACTION_HINT_KEYWORDS:
            if keyword in lowered:
                score += 3
        if index < 2:
            score += 2
        scored.append((score, index, chunk.get("page", 0), text))

    scored.sort(key=lambda item: (-item[0], item[1]))
    passages = []
    seen = set()
    for _, _, page, text in scored:
        snippet = text[:900]
        key = snippet.lower()
        if key in seen:
            continue
        seen.add(key)
        passages.append(f"[p.{page}] {snippet}")
        if len(passages) >= limit:
            break
    return passages


def build_extraction_prompt(title: str, abstract: str, passages: list[str]) -> str:
    passage_block = "\n".join(f"- {item}" for item in passages) if passages else "- (no extra passages selected)"
    return (
        f"Paper title:\n{title}\n\n"
        f"Abstract:\n{abstract}\n\n"
        f"Relevant passages:\n{passage_block}\n\n"
        "Extract the structured metadata now."
    )


def infer_paper_type(text: str) -> str:
    lowered = text.lower()
    if any(keyword in lowered for keyword in ("survey", "review")):
        return "survey"
    if any(keyword in lowered for keyword in ("benchmark", "evaluation", "dataset")):
        return "benchmark"
    if any(keyword in lowered for keyword in ("application", "case study", "use case")):
        return "application"
    if any(keyword in lowered for keyword in ("system", "framework", "pipeline")):
        return "system"
    if any(keyword in lowered for keyword in ("theorem", "theoretical", "proof")):
        return "theory"
    return "method"


TASK_RULES = (
    ("node classification", "node classification"),
    ("graph classification", "graph classification"),
    ("link prediction", "link prediction"),
    ("recommendation", "recommendation"),
    ("question answering", "question answering"),
    ("qa", "question answering"),
    ("summarization", "summarization"),
    ("retrieval", "retrieval"),
    ("reasoning", "reasoning"),
    ("molecular property prediction", "molecular property prediction"),
    ("property prediction", "property prediction"),
    ("entity linking", "entity linking"),
    ("information extraction", "information extraction"),
    ("clustering", "clustering"),
    ("classification", "classification"),
)

APPLICATION_RULES = (
    ("biomedical", "biomedical"),
    ("drug discovery", "drug discovery"),
    ("molecule", "molecular analysis"),
    ("chemistry", "chemistry"),
    ("traffic", "traffic forecasting"),
    ("recommender", "recommender systems"),
    ("social network", "social networks"),
    ("knowledge graph", "knowledge graphs"),
    ("legal", "legal documents"),
    ("finance", "financial analysis"),
    ("code", "code understanding"),
    ("document", "document analysis"),
    ("search", "search"),
)

DATASET_RULES = (
    "cora",
    "citeseer",
    "pubmed",
    "ogb",
    "reddit",
    "arxiv",
    "amazon",
    "dblp",
    "wiki",
    "movielens",
    "yelp",
)


def rule_based_extraction(title: str, abstract: str, passages: list[str]) -> dict[str, Any]:
    text = " ".join([title, abstract, " ".join(passages)]).lower()
    tasks = [label for pattern, label in TASK_RULES if pattern in text]
    applications = [label for pattern, label in APPLICATION_RULES if pattern in text]
    datasets = []
    for pattern in DATASET_RULES:
        if pattern in text:
            datasets.append(pattern.upper() if pattern == "ogb" else pattern.title())

    if any(keyword in text for keyword in ("survey", "review")):
        paper_type = "survey"
    elif any(keyword in text for keyword in ("benchmark", "evaluation")):
        paper_type = "benchmark"
    elif applications:
        paper_type = "application"
    else:
        paper_type = infer_paper_type(text)

    return {
        "paper_type": paper_type,
        "tasks": list(dict.fromkeys(tasks)),
        "applications": list(dict.fromkeys(applications)),
        "datasets": list(dict.fromkeys(datasets)),
        "method_summary": "",
        "contribution_summary": "",
        "confidence": "low" if not (tasks or applications or datasets) else "medium",
    }


def extract_structured_metadata(title: str, abstract: str, raw_text: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    passages = select_extraction_passages(chunks)
    prompt = build_extraction_prompt(title, abstract, passages)
    fallback_text = " ".join([title, abstract, " ".join(passages)])

    try:
        raw = call_llm(
            prompt=prompt,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=1200,
        )
        parsed = parse_json_response(raw)
        if isinstance(parsed, dict):
            print(f"[LLM] extracted structured metadata for: {title}", flush=True)
            result = {
                "paper_type": str(parsed.get("paper_type", infer_paper_type(" ".join([title, abstract, raw_text])))).strip() or "other",
                "tasks": normalize_list(parsed.get("tasks", [])),
                "applications": normalize_list(parsed.get("applications", [])),
                "datasets": normalize_list(parsed.get("datasets", [])),
                "method_summary": normalize_text(str(parsed.get("method_summary", ""))),
                "contribution_summary": normalize_text(str(parsed.get("contribution_summary", ""))),
                "confidence": str(parsed.get("confidence", "low")).strip() or "low",
            }
            if not result["paper_type"]:
                result["paper_type"] = infer_paper_type(fallback_text)
            if not result["tasks"] and not result["applications"] and not result["datasets"]:
                fallback = rule_based_extraction(title, abstract, passages)
                for key in ("paper_type", "tasks", "applications", "datasets"):
                    if not result.get(key):
                        result[key] = fallback[key]
                if not result["method_summary"]:
                    result["method_summary"] = fallback["method_summary"]
                if not result["contribution_summary"]:
                    result["contribution_summary"] = fallback["contribution_summary"]
                if result.get("confidence", "low") == "low":
                    result["confidence"] = fallback["confidence"]
            return result
    except Exception:
        pass

    print(f"[RULE] fallback metadata for: {title}", flush=True)
    return rule_based_extraction(title, abstract, passages)


def build_cache_key(pdf_path: Path) -> str:
    stat = pdf_path.stat()
    return f"{pdf_path.name}:{stat.st_mtime_ns}:{stat.st_size}"


def build_chunks_from_pages(file_name: str, title: str, page_texts: list[str]) -> list[dict[str, Any]]:
    chunks = []
    chunk_id = 0

    for page_number, page_text in enumerate(page_texts, start=1):
        normalized_page = normalize_text(page_text)
        if not normalized_page:
            continue

        paragraphs = [normalize_text(part) for part in re.split(r"\n\s*\n", page_text) if normalize_text(part)]
        if not paragraphs:
            paragraphs = [normalized_page]

        buffer = ""
        for paragraph in paragraphs:
            if len(buffer) + len(paragraph) + 1 <= 900:
                buffer = f"{buffer} {paragraph}".strip()
                continue

            if buffer:
                chunks.append(
                    {
                        "chunk_id": f"{file_name}::chunk::{chunk_id}",
                        "file_name": file_name,
                        "title": title,
                        "page": page_number,
                        "text": buffer,
                    }
                )
                chunk_id += 1
            buffer = paragraph

        if buffer:
            chunks.append(
                {
                    "chunk_id": f"{file_name}::chunk::{chunk_id}",
                    "file_name": file_name,
                    "title": title,
                    "page": page_number,
                    "text": buffer,
                }
            )
            chunk_id += 1

    return chunks


def read_pdf(state: PaperState) -> dict:
    if fitz is None:
        raise ImportError("Missing PyMuPDF. Install it with: pip install pymupdf")

    pdf_path = Path(state["pdf_path"])
    fallback_title = pdf_path.stem

    page_texts = []
    with fitz.open(pdf_path) as doc:
        title = extract_title_from_first_page(doc, fallback_title)
        for page_index in range(len(doc)):
            page_texts.append(doc.load_page(page_index).get_text("text"))

    raw_text = "\n".join(page_texts)
    chunks = build_chunks_from_pages(pdf_path.name, title, page_texts)
    return {"title": title, "raw_text": raw_text, "chunks": chunks}


def extract_abstract(state: PaperState) -> dict:
    text = normalize_text(state["raw_text"])
    match = ABSTRACT_PATTERN.search(text)
    abstract = match.group(1).strip() if match else ""
    if not abstract:
        abstract = text[:1500].strip()
    return {"abstract": abstract}


def generate_tags(state: PaperState) -> dict:
    text = f"{state['title']} {state['abstract']}".lower()
    tags = []
    for keyword, tag in KEYWORD_RULES.items():
        if keyword in text and tag not in tags:
            tags.append(tag)
    if not tags:
        tags.append("Graph Learning")
    return {"tags": tags, "category": infer_category(text)}


def build_graph():
    builder = StateGraph(PaperState)
    builder.add_node("read_pdf", read_pdf)
    builder.add_node("extract_abstract", extract_abstract)
    builder.add_node("generate_tags", generate_tags)
    builder.add_edge(START, "read_pdf")
    builder.add_edge("read_pdf", "extract_abstract")
    builder.add_edge("extract_abstract", "generate_tags")
    builder.add_edge("generate_tags", END)
    return builder.compile()


def process_papers() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    graph = build_graph()
    summaries = []
    all_chunks = []
    metadata_cache = load_json_file(METADATA_CACHE_PATH, {})

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    print(f"Processing {len(pdf_files)} papers...", flush=True)

    for index, pdf_path in enumerate(pdf_files, start=1):
        print(f"[{index}/{len(pdf_files)}] {pdf_path.name}", flush=True)
        initial_state: PaperState = {
            "pdf_path": str(pdf_path),
            "title": "",
            "raw_text": "",
            "abstract": "",
            "tags": [],
            "category": "",
            "chunks": [],
        }
        result = graph.invoke(initial_state)
        cache_key = build_cache_key(pdf_path)
        cached_metadata = metadata_cache.get(cache_key)
        if isinstance(cached_metadata, dict):
            metadata = cached_metadata
        else:
            metadata = extract_structured_metadata(
                result["title"],
                result["abstract"],
                result["raw_text"],
                result["chunks"],
            )
            metadata_cache[cache_key] = metadata
        summaries.append(
            {
                "file_name": pdf_path.name,
                "title": result["title"],
                "abstract": result["abstract"],
                "tags": result["tags"],
                "category": result["category"],
                "paper_type": metadata["paper_type"],
                "tasks": metadata["tasks"],
                "applications": metadata["applications"],
                "datasets": metadata["datasets"],
                "method_summary": metadata["method_summary"],
                "contribution_summary": metadata["contribution_summary"],
                "confidence": metadata["confidence"],
            }
        )
        all_chunks.extend(result["chunks"])

    return summaries, all_chunks, metadata_cache


def save_results(summaries: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(exist_ok=True)
    save_json_file(SUMMARY_PATH, summaries)
    save_json_file(CHUNKS_PATH, chunks)
    return SUMMARY_PATH, CHUNKS_PATH


def main() -> None:
    summaries, chunks, metadata_cache = process_papers()
    summary_path, chunk_path = save_results(summaries, chunks)
    save_json_file(METADATA_CACHE_PATH, metadata_cache)
    print(f"Processed {len(summaries)} papers")
    print(f"Saved summaries to: {summary_path}")
    print(f"Saved chunks to: {chunk_path}")


if __name__ == "__main__":
    main()
