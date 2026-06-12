"""
agent/agent_graph.py — LangGraph ReAct Agent
=============================================
这是整个项目的核心：用 LangGraph 的状态图实现 ReAct Agent 循环。

ReAct 循环（面试必问）：
  观察(Observe) → 思考(Think) → 行动(Act) → 观察(Observe) → ...

图结构：
  [开始] → [LLM 推理] → 判断：直接回答？→ [结束]
                           ↓ 需要工具？
                        [执行工具] → 回到 [LLM 推理]

LangGraph 三要素：
  State  — Agent 的"记忆"，在节点间流转的数据
  Node   — 图中的"步骤"（调用 LLM / 执行工具）
  Edge   — 节点之间的"箭头"（普通边 / 条件边）
"""

from typing import Annotated, List, Literal, Optional
from typing_extensions import TypedDict
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .llm import get_llm
from .tools import ALL_TOOLS
from .rag import (
    load_vectorstore,
    retrieve,
    format_retrieved_context,
    get_embeddings,
)
from .memory import get_memory


# ============================================================
# State 定义 — Agent 的状态在节点间流转
# ============================================================
class AgentState(TypedDict):
    """
    Agent 的状态快照。每次经过一个节点都可能被更新。
    messages: 消息历史（用 operator.add 累加，不是替换）
    """
    messages: Annotated[List[BaseMessage], operator.add]
    context: str  # RAG 检索到的上下文


# ============================================================
# 绑定工具到 LLM
# ============================================================
def _bind_tools():
    """把工具列表绑定到 LLM，让 LLM '知道'它可以调用哪些工具"""
    llm = get_llm(temperature=0.3)  # Agent 推理用低温度，减少随机性
    return llm.bind_tools(ALL_TOOLS)


# ============================================================
# Node 1: LLM 推理节点（Agent 的"大脑"）
# ============================================================
def agent_node(state: AgentState) -> dict:
    """
    Agent 推理步骤：
    1. 拿当前状态里的消息
    2. 如果有 RAG 上下文，拼到 system prompt 里
    3. 调用 LLM（带着工具列表）
    4. LLM 返回：要么直接回答，要么说"我要调用某某工具"

    面试考点：这一步就是 ReAct 中的 "Reasoning"
    """
    messages = state["messages"]
    context = state.get("context", "")

    # 如果有 RAG 检索到的文档，塞进 system message
    if context:
        system_prompt = (
            "你是一个智能助手，可以调用工具来完成任务。\n\n"
            "以下是用户上传的文档中与问题相关的内容，请基于这些内容回答：\n\n"
            f"{context}\n\n"
            "如果文档内容不够回答问题，请如实告知并尝试用其他工具搜索。"
        )
        # 把 system prompt 放在最前面
        from langchain_core.messages import SystemMessage

        full_messages = [SystemMessage(content=system_prompt)] + list(messages)
    else:
        system_prompt = (
            "你是一个智能助手，可以调用工具来完成任务。"
            "优先使用工具获取实时信息，再给出准确回答。"
        )
        from langchain_core.messages import SystemMessage

        full_messages = [SystemMessage(content=system_prompt)] + list(messages)

    llm_with_tools = _bind_tools()
    response = llm_with_tools.invoke(full_messages)

    return {"messages": [response]}


# ============================================================
# Node 2: 工具执行节点（Agent 的"手脚"）
# ============================================================
def tool_node(state: AgentState) -> dict:
    """
    执行 LLM 要求的工具调用。

    面试考点：这就是 ReAct 中的 "Acting"
    LLM 说"我要查天气"→ 这个节点真正执行 get_weather() → 把结果包装成 ToolMessage 返回

    流程：
    1. 从最后一条 AI 消息中提取 tool_calls
    2. 逐一执行工具
    3. 把执行结果以 ToolMessage 形式追加到消息流
    """
    messages = state["messages"]
    last_message = messages[-1]

    # 构建工具名 → 工具对象的映射
    tool_map = {tool.name: tool for tool in ALL_TOOLS}

    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        print(f"🔧 [Agent] 调用工具：{tool_name}({tool_args})")

        if tool_name in tool_map:
            try:
                result = tool_map[tool_name].invoke(tool_args)
                print(f"📦 [Agent] 工具返回：{str(result)[:200]}...")
            except Exception as e:
                result = f"工具执行出错：{str(e)}"
                print(f"❌ [Agent] 工具出错：{e}")
        else:
            result = f"未知工具：{tool_name}"

        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )

    return {"messages": tool_messages}


# ============================================================
# 路由判断：继续还是结束？
# ============================================================
def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """
    判断 Agent 下一步该走哪个节点。
    - 如果 LLM 的最后一条消息包含 tool_calls → 去执行工具
    - 如果没有 tool_calls → Agent 已经给出最终回答，结束

    面试考点：这是 ReAct 循环的"终止条件"
    """
    messages = state["messages"]
    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "__end__"


# ============================================================
# 构建 LangGraph 图
# ============================================================
def build_agent_graph():
    """
    搭建 ReAct Agent 的状态图。

    图结构：
      START → agent（LLM 推理）
                ↓
          should_continue?
           ↙           ↘
      tools（执行工具）  END（结束）
           ↘           ↗
            back to agent
    """
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    # 设置入口：从 agent 节点开始
    workflow.set_entry_point("agent")

    # 条件边：agent 之后根据 should_continue 决定去向
    workflow.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        "__end__": END,
    })

    # 普通边：工具执行完 → 回到 agent 继续推理
    workflow.add_edge("tools", "agent")

    # 编译（带记忆——LangGraph 自动管理对话历史）
    memory_checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=memory_checkpointer)

    return app


# ============================================================
# RAG 增强的 Agent 运行器
# ============================================================
_agent_app = None
_rag_vectorstore = None


def get_agent_app():
    """获取全局 Agent 实例（单例）"""
    global _agent_app
    if _agent_app is None:
        _agent_app = build_agent_graph()
    return _agent_app


def init_rag(knowledge_base_dir: str = "./data/chroma_db"):
    """初始化 RAG 向量数据库"""
    global _rag_vectorstore
    try:
        from .rag import is_rag_available
        if not is_rag_available():
            print("⚠️  RAG 不可用（embedding 模型未下载），Agent 工具调用功能正常")
            return
        _rag_vectorstore = load_vectorstore(knowledge_base_dir)
        if _rag_vectorstore is not None:
            print(f"📚 RAG 已就绪，向量库路径：{knowledge_base_dir}")
        else:
            print("⚠️  RAG 向量库为空，上传文档后将自动创建")
    except Exception as e:
        print(f"⚠️  RAG 初始化跳过：{e}")


def run_agent(
    user_input: str,
    use_rag: bool = True,
    thread_id: str = "default",
) -> str:
    """
    运行 Agent 的主入口。

    参数：
      user_input: 用户输入的自然语言
      use_rag: 是否启用 RAG 检索
      thread_id: 对话线程 ID（不同线程有独立的记忆）

    面试考点：这就是完整的 ReAct 循环：
      用户输入 → RAG检索 → Agent推理 → 工具调用 → 再推理 → 输出
    """
    global _rag_vectorstore

    app = get_agent_app()
    memory = get_memory()

    # 1. RAG 检索（如果启用且有向量库）
    context = ""
    if use_rag and _rag_vectorstore is not None:
        try:
            results = retrieve(user_input, _rag_vectorstore, top_k=3)
            context = format_retrieved_context(results)
            if results:
                print(f"📚 [RAG] 检索到 {len(results)} 条相关内容")
        except Exception as e:
            print(f"⚠️  [RAG] 检索失败：{e}")

    # 2. 构建初始状态
    memory.add_user_message(user_input)
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "context": context,
    }

    # 3. 运行 Agent 图，直到结束
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = app.invoke(initial_state, config)
        # 提取最后一条 AI 消息
        final_message = result["messages"][-1]
        answer = final_message.content

        memory.add_ai_message(answer)
        return answer

    except Exception as e:
        error_msg = f"❌ Agent 运行出错：{str(e)}"
        print(error_msg)
        return error_msg


# ============================================================
# 重新索引文档（给 RAG 上传新文档时调用）
# ============================================================
def reindex_documents(file_paths: List[str], persist_dir: str = "./data/chroma_db"):
    """上传新文档 → 加载 → 分块 → 向量化 → 存入向量库"""
    from .rag import load_document, split_documents, create_vectorstore, is_rag_available

    global _rag_vectorstore

    if not is_rag_available():
        raise RuntimeError("RAG 不可用：embedding 模型未下载，无法索引文档")

    all_chunks = []
    for path in file_paths:
        docs = load_document(path)
        chunks = split_documents(docs)
        all_chunks.extend(chunks)

    _rag_vectorstore = create_vectorstore(all_chunks, persist_dir)
    if _rag_vectorstore is None:
        raise RuntimeError("向量库创建失败：embedding 模型不可用")
    return len(all_chunks)
