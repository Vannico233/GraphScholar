# Day 3 Demo Outputs

下面是 `python run_agent.py --demo --no-llm` 的 5 组示例输出摘要版。

## Q1

问题：

```text
GraphRAG 方向有哪些代表性论文？
```

示例输出要点：

- 代表工作：
  - `A Survey of Graph Retrieval-Augmented Generation`
  - `GraphRAG-Bench`
  - `Retrieval-Augmented Generation with Graphs (GraphRAG)`
- 核心方法：
  - GraphRAG 通过图结构组织知识与检索证据
  - survey 类论文总结了图表示、图检索、多跳推理、结构化生成
- 局限：
  - 目前检索主要依赖关键词重叠
  - 当前总结主要来自摘要

## Q2

问题：

```text
GraphRAG benchmark 相关论文有哪些？
```

示例输出要点：

- 代表工作：
  - `GraphRAG-Bench`
  - `LinearRAG`
  - `Retrieval-Augmented Generation with Graphs (GraphRAG)`
- 核心方法：
  - benchmark / evaluation 关注 GraphRAG 是否真正优于普通 RAG
  - 重点比较结构化图检索与普通文本检索的差异

## Q3

问题：

```text
Knowledge Graph RAG 方向有哪些代表工作？
```

示例输出要点：

- 代表工作：
  - `Knowledge Graph-Guided Retrieval Augmented Generation (KG²RAG)`
  - `From Local to Global A Graph RAG Approach to Query-Focused Summarization`
  - `GraphRAG-Bench`
- 核心方法：
  - 用知识图谱组织实体关系
  - 改进检索与证据组织
  - 支持更复杂的多跳与全局问题回答

## Q4

问题：

```text
图对比学习有哪些经典论文？
```

示例输出要点：

- 代表工作：
  - `GCC`
  - `GRACE`
  - `HGMAE`
- 核心方法：
  - 通过图增广和对比目标学习表示
  - 强调自监督、无标签训练、表示一致性

## Q5

问题：

```text
图预训练和图提示学习有哪些代表工作？
```

示例输出要点：

- 代表工作：
  - `All in One Multi-task Prompting for Graph Neural Networks`
  - `GCC`
  - `FUG`
- 核心方法：
  - 图预训练强调迁移泛化
  - 图提示学习尝试缩小预训练目标与下游任务之间的差距

## 说明

这些输出是当前 Day 3 系统的稳定演示版本，重点在于：

- 能跑通完整主链路
- 能稳定返回结构化结果
- 适合课程展示与后续迭代

如果启用 LLM 模式，最终答案会由 `src/agent.py` 中的单智能体结合检索结果进一步组织生成。
