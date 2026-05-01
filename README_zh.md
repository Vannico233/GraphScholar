<table>
  <tr>
    <td><h1>GraphScholar</h1></td>
    <td align="right"><a href="./README.md">English</a> / 中文</td>
  </tr>
</table>

![GraphScholar](GraphScholar.png)

GraphScholar 是一个面向图学习和 GraphRAG 论文分析的研究助手。它会把本地 PDF 论文整理成结构化元数据、chunk 级证据和基于图谱的文献库，用于检索和研究型问答。

## 项目功能

- 读取 `data/` 中的 PDF 论文
- 抽取标题、摘要、任务、应用场景、数据集、方法摘要和贡献摘要
- 生成论文级摘要和全文 chunk 证据
- 构建 paper-topic-method-task-application-dataset 图谱
- 支持检索、图查询和结构化问答
- 每次运行都会保存到 `outputs/answer/`

## 主要输出

- `outputs/paper_summaries.json`
- `outputs/paper_chunks.json`
- `outputs/paper_graph.json`
- `outputs/paper_metadata_cache.json`
- `outputs/answer/*.md`

## 工作流程

1. `paper_organization.py` 读取 PDF 并抽取结构化信息。
2. `src/build_graph.py` 根据摘要构建论文图谱。
3. `src/tools.py` 负责论文检索、chunk 检索和图查询。
4. `src/agent.py` 负责理解问题、收集证据并生成回答。
5. `run_agent.py` 运行 demo 或单问题模式，并写出报告。

## 数据结构

每篇论文摘要包含：

- `title`
- `abstract`
- `tags`
- `category`
- `paper_type`
- `tasks`
- `applications`
- `datasets`
- `method_summary`
- `contribution_summary`
- `confidence`

## 如何运行

重新整理论文库：

```powershell
python paper_organization.py
```

本地确定性模式：

```powershell
python run_agent.py --no-llm
```

回答单个问题：

```powershell
python run_agent.py --question "如果我在做 GraphRAG，请帮我梳理近几年最有代表性的论文，并按方法、评测和综述分类。"
```

运行预设 demo：

```powershell
python run_agent.py --demo
```

## LLM 配置

`src/llm_client.py` 使用显式代码内配置：

- `DEFAULT_BASE_URL`
- `DEFAULT_API_KEY`
- `DEFAULT_MODEL_ID`

使用 LLM 模式前，请先填写这三个值。

## 项目特点

它不是一个普通的关键词搜索工具，而是结合了：

- 结构化论文抽取
- 图增强文献组织
- 面向任务、应用和数据集的检索
- 带页码的 chunk 证据
- 可保存的问答报告

## 说明

这个项目整体比较轻量，但已经可以作为一个完整的图学习文献分析工作流来使用。
