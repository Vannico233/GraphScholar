# Day 3 Preset Questions

下面 5 组问题用于验证 Day 3 的单智能体研究助手主流程是否稳定。

## Q1

问题：

```text
GraphRAG 方向有哪些代表性论文？
```

预期命中：

- `A Survey of Graph Retrieval-Augmented Generation`
- `Graph Retrieval-Augmented Generation A Survey`
- `Retrieval-Augmented Generation with Graphs (GraphRAG)`
- `From Local to Global A Graph RAG Approach to Query-Focused Summarization`

输出重点：

- 区分 survey 与方法类论文
- 点出 GraphRAG 的图结构知识表示、多跳检索、结构化生成

## Q2

问题：

```text
GraphRAG benchmark 相关论文有哪些？
```

预期命中：

- `GraphRAG-Bench`
- `When to use Graphs in RAG`

输出重点：

- benchmark / evaluation / 何时使用图结构
- 评价 GraphRAG 是否真的优于普通 RAG

## Q3

问题：

```text
Knowledge Graph RAG 方向有哪些代表工作？
```

预期命中：

- `Knowledge Graph-Guided Retrieval Augmented Generation (KG²RAG)`
- `Retrieval-Augmented Generation with Graphs (GraphRAG)`

输出重点：

- 知识图谱如何帮助检索与组织证据
- 与普通文本 RAG 的差别

## Q4

问题：

```text
图对比学习有哪些经典论文？
```

预期命中：

- `GCC`
- `GRACE`
- `GCA`
- `BGRL`
- `FUG`

输出重点：

- 对比学习目标
- augmentation 设计
- 自监督图表示学习能力

## Q5

问题：

```text
图预训练和图提示学习有哪些代表工作？
```

预期命中：

- `All in One Multi-task Prompting for Graph Neural Networks`
- `GCC`
- `FUG`

输出重点：

- 图预训练与下游任务迁移
- 图提示学习如何缩小预训练与多任务之间的 gap

## 建议运行方式

单问题：

```powershell
python run_agent.py --question "GraphRAG 方向有哪些代表性论文？"
```

5 组预设问题：

```powershell
python run_agent.py --demo
```

如果暂时不想调用大模型，可以先使用稳定的本地模板输出：

```powershell
python run_agent.py --demo --no-llm
```
