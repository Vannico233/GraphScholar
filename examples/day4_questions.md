# Day 4 Graph-Aware Questions

下面 5 组问题用于验证 Day 4 的图查询能力。

## Q1

```text
GraphRAG 主题下常见方法有哪些？
```

目标：

- 走图查询分支
- 返回主题下常见方法和代表论文

## Q2

```text
哪些论文连接了 GraphRAG 和 Knowledge Graph？
```

目标：

- 走桥接论文查询
- 找到同时连接两个主题的论文

## Q3

```text
Graph Contrastive Learning 主题下有哪些代表方法？
```

目标：

- 从 topic -> method 查询代表方法

## Q4

```text
Graph Pretraining 和 Graph Representation Learning 之间有哪些连接论文？
```

目标：

- 找桥接两个主题的论文

## Q5

```text
Graph Prompting 最邻近的主题有哪些？
```

目标：

- 找到通过共享论文形成连接的邻近主题

## 建议运行方式

单问题：

```powershell
python run_agent.py --question "GraphRAG 主题下常见方法有哪些？" --no-llm
```

批量测试：

```powershell
python run_agent.py --demo --no-llm
```

如果你想把 Day 4 的 5 个问题单独跑，可以把 `DAY4_PRESET_QUESTIONS` 复制到 `run_agent.py` 中临时替换 `PRESET_QUESTIONS`，或者直接逐条运行。
