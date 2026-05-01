import argparse
import sys
from datetime import datetime
from pathlib import Path

from src.agent import ResearchAssistantAgent


PRESET_QUESTIONS = [
    "如果我在做 GraphRAG，请帮我梳理近几年最有代表性的论文，并按方法、评测和综述分类。",
    "围绕 GraphRAG benchmark，哪些论文更适合用来评估检索、推理和全流程生成能力？",
    "如果我研究 Knowledge Graph RAG，请推荐几篇能体现知识图谱如何参与检索和组织证据的论文。",
    "图对比学习里，哪些论文在 node classification 和 graph classification 上表现更好，适合做近期代表作汇总？",
    "图预训练和图提示学习之间的关系是什么？请推荐几篇能体现从 pretraining 到 prompting 迁移的论文。",
    "如果我关注 link prediction 这个下游任务，请找一些图对比学习方向里表现强、机制也比较清楚的论文。",
    "如果我研究谱方法在图学习中的应用，请推荐一些能支撑架构设计或高低频建模的代表论文。",
    "对于 heterophily graph learning，有哪些论文从结构建模或表征学习角度给出了比较有说服力的方案？",
    "如果我要把图学习机制迁移到推荐系统或知识图谱问答里，哪些论文最值得优先看？",
    "请帮我找一些既有方法贡献、又有明确实验任务和应用场景的图学习论文，适合拿来做项目背景梳理。",
]


def format_search_block(result) -> str:
    lines = ["候选论文"]
    for item in result.search_results:
        extra = []
        if item.get("tasks"):
            extra.append(f"tasks={', '.join(item['tasks'])}")
        if item.get("applications"):
            extra.append(f"apps={', '.join(item['applications'])}")
        if item.get("datasets"):
            extra.append(f"dsets={', '.join(item['datasets'])}")
        if item.get("paper_type"):
            extra.append(f"type={item['paper_type']}")
        extra_text = f" | {' | '.join(extra)}" if extra else ""
        lines.append(
            f"- {item['title']} | category={item['category']} | score={item['score']} | tags={', '.join(item['tags'])}{extra_text}"
        )
    return "\n".join(lines)


def build_run_report(question_items) -> str:
    lines = ["# Run Report", ""]
    for index, item in enumerate(question_items, start=1):
        result = item["result"]
        lines.extend(
            [
                f"## Question {index}",
                f"**Q:** {item['question']}",
                f"**route_type:** {result.route_type}",
                "",
                "### Candidates",
                format_search_block(result),
                "",
                "### Answer",
                result.final_answer,
                "",
            ]
        )
    return "\n".join(lines)


def save_run_report(report_text: str) -> Path:
    output_dir = Path("outputs") / "answer"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"run_{timestamp}.md"
    report_path.write_text(report_text, encoding="utf-8")
    return report_path

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Day 3 graph literature research assistant")
    parser.add_argument("--question", type=str, help="Single research question to ask")
    parser.add_argument("--no-llm", action="store_true", help="Use deterministic formatting only")
    parser.add_argument("--demo", action="store_true", help="Run 10 preset questions")
    args = parser.parse_args()

    agent = ResearchAssistantAgent(use_llm=not args.no_llm)

    questions = []
    if args.question:
        questions = [args.question]
    elif args.demo:
        questions = PRESET_QUESTIONS
    else:
        questions = PRESET_QUESTIONS

    run_items = []
    for index, question in enumerate(questions, start=1):
        result = agent.run(question)
        run_items.append({"question": question, "result": result})
        print(f"===== Question {index} =====")
        print(question)
        print(f"route_type: {result.route_type}")
        print()
        print(format_search_block(result))
        print()
        print(result.final_answer)
        print()

    report_path = save_run_report(build_run_report(run_items))
    print(f"Saved run report to: {report_path}")


if __name__ == "__main__":
    main()
