# 🤖 多工具 ReAct Agent

基于 **DeepSeek + LangGraph** 的多工具智能 Agent，支持自主决策调用工具、联网搜索、数学计算、RAG 文档检索、对话记忆。

🔗 **Live Demo**: 本地运行 `streamlit run main.py`

---

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| 🧠 **ReAct Agent** | LangGraph 状态图实现，Agent 自主「观察→思考→行动」循环 |
| 🛠️ **多工具调用** | 计算器、联网搜索、日期时间，LLM 自动判断该用哪个 |
| 📚 **RAG 文档检索** | 上传 PDF/TXT → 自动向量化 → 基于文档内容精准问答 |
| 🧩 **对话记忆** | 两层记忆架构：MemorySaver（状态图层）+ 滑动窗口（防爆上下文） |
| 🖥️ **Streamlit 前端** | 纯 Python Web 界面，即开即用 |
| 🐳 **Docker 部署** | 一行命令上线 |

---

## 🏗️ 架构

```
用户输入 → Streamlit 前端
              ↓
         LangGraph Agent 图
         ┌─────────────────┐
         │  agent_node      │ ← LLM 推理（DeepSeek）
         │   ↓ 需要工具？    │
         │  tool_node        │ ← 执行工具调用
         │   ↓ 结果喂回      │
         │  agent_node       │ ← 再推理 → 输出
         └─────────────────┘
              ↓
         RAG 检索（可选）
         文档 → 分块 → 向量化 → 检索 → 拼接 Prompt
```

## 🔧 ReAct 循环

```
START → [agent: LLM推理] → 有 tool_call? → [tools: 执行工具] → 回到 agent
                  ↓ 无 tool_call
                 END（返回结果）
```

---

## 🚀 快速开始

### 1. 安装

```bash
git clone git@github.com:nineteem526/agent-.git
cd agent-
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入 DeepSeek API Key（https://platform.deepseek.com）
```

### 3. 启动

```bash
streamlit run main.py
```

浏览器打开 `http://localhost:8501`

### Docker 一键部署

```bash
docker build -t agent-app -f docker/Dockerfile .
docker run -p 8501:8501 --env-file .env agent-app
```

---

## 🧪 试试这些

```
帮我算 sqrt(144) + 3 * 5        → 测试工具调用（calculator）
搜索最新的 AI Agent 新闻          → 测试联网搜索（web_search）
今天几号？周几？                  → 测试时间工具
[上传 PDF] → 这篇文章讲了什么？    → 测试 RAG 文档检索
我叫张三 → 我叫什么名字？          → 测试对话记忆
```

---

## 📁 项目结构

```
agent-project/
├── main.py                  # Streamlit 前端入口
├── requirements.txt         # Python 依赖
├── .env.example             # API Key 模板
│
├── agent/                   # 核心逻辑
│   ├── llm.py               #   DeepSeek API 封装
│   ├── tools.py             #   工具定义（计算器/搜索/时间）
│   ├── agent_graph.py       #   LangGraph ReAct Agent 状态图
│   ├── rag.py               #   RAG 文档检索系统
│   └── memory.py            #   对话记忆管理
│
├── docker/                  # 容器化部署
│   ├── Dockerfile
│   └── docker-compose.yml
│
└── data/                    # 向量数据库存储
```

---

## 🛠️ 技术栈

`Python` `LangChain` `LangGraph` `DeepSeek API` `Streamlit` `ChromaDB` `Docker` `Function Calling` `RAG` `ReAct`

---

## 📝 面试要点

本项目覆盖 AI Agent 实习岗位的核心考察点：

- **Function Calling**: `tools.py` — `@tool` 装饰器，LLM 决策 + Python 执行
- **ReAct 循环**: `agent_graph.py` — State / Node / Edge 三要素，`should_continue` 终止条件
- **对话记忆**: `memory.py` — MemorySaver + 滑动窗口（max_turns=20）
- **RAG 管线**: `rag.py` — 文档加载→分块→向量化→检索→拼接 Prompt

---

## 👤 作者

- GitHub: [nineteem526](https://github.com/nineteem526)
- 项目周期：2025.06 · 独立开发
