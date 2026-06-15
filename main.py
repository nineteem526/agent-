"""
main.py — Streamlit 前端入口
=============================
纯 Python 的 Web 界面，无需前端知识。
运行：streamlit run main.py
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="AI Agent · 多工具助手",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 多工具 ReAct Agent")
st.caption("基于 DeepSeek + LangGraph · 支持 RAG 文档问答 · 工具调用 · 对话记忆")

# ============================================================
# 初始化 Agent（只初始化一次）
# ============================================================
@st.cache_resource
def init_agent():
    """缓存 Agent 实例，避免每次刷新重新创建"""
    from agent.agent_graph import get_agent_app
    return get_agent_app()


with st.spinner("🚀 Agent 启动中..."):
    app = init_agent()
    from agent.rag import is_rag_available
    rag_status = is_rag_available()  # 不触发下载，只检查状态

# ============================================================
# 侧边栏：文档上传 & 工具状态
# ============================================================
with st.sidebar:
    st.header("📂 文档上传（RAG）")

    uploaded_files = st.file_uploader(
        "上传 PDF / TXT / MD 文件",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="上传后 Agent 可以基于这些文档回答问题",
    )

    if uploaded_files and st.button("📥 索引文档", type="primary"):
        with st.spinner("文档处理中：加载 → 分块 → 向量化..."):
            from agent.agent_graph import reindex_documents, init_rag
            from agent.rag import is_rag_available

            # 只在用户真正上传文档时才尝试初始化 RAG
            if not is_rag_available():
                try:
                    from agent.rag import get_embeddings
                    emb = get_embeddings()
                    if emb is None:
                        st.error("❌ Embedding 模型下载失败，RAG 不可用。请检查网络后重试。")
                        st.stop()
                except Exception as e:
                    st.error(f"❌ RAG 初始化失败：{e}")
                    st.stop()

            temp_dir = tempfile.mkdtemp()
            file_paths = []
            for f in uploaded_files:
                path = os.path.join(temp_dir, f.name)
                with open(path, "wb") as wf:
                    wf.write(f.getbuffer())
                file_paths.append(path)

            try:
                chunk_count = reindex_documents(file_paths, "./data/chroma_db")
                st.success(f"✅ 已索引 {len(uploaded_files)} 个文档，共 {chunk_count} 个文本块")
            except Exception as e:
                st.error(f"索引失败：{e}")

    st.divider()

    st.header("🛠️ 可用工具")
    st.markdown("""
    | 工具 | 功能 |
    |------|------|
    | 🧮 `calculator` | 数学计算（支持 sqrt/sin/cos 等） |
    | 🔍 `web_search` | 网络搜索（获取实时信息） |
    | 🕐 `get_current_time` | 当前日期时间 |

    *Agent 自主判断该用哪个工具*
    """)

    st.divider()

    st.divider()

    st.header("📊 状态")
    if not rag_status:
        st.warning("⚠️ RAG 未就绪")

        if st.button("🔧 初始化 RAG（下载 embedding 模型）", type="primary"):
            with st.spinner("正在下载 embedding 模型（约 400MB，仅首次需要）..."):
                from agent.rag import get_embeddings
                emb = get_embeddings()
                if emb is not None:
                    st.cache_resource.clear()
                    st.success("✅ RAG 已就绪！现在可以上传文档了")
                    st.rerun()
                else:
                    st.error("❌ 下载失败，请检查网络后重试")
        use_rag = False
    else:
        use_rag = st.toggle("启用 RAG 检索", value=True)
        st.success("✅ RAG 已就绪")
    if st.button("🗑️ 清除对话记忆"):
        from agent.memory import get_memory

        get_memory().clear()
        st.session_state.messages = []
        st.rerun()

# ============================================================
# 主区域：对话界面
# ============================================================
# 初始化消息历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 输入框
if prompt := st.chat_input("输入你的问题，Agent 会自主决定怎么回答..."):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用 Agent
    with st.chat_message("assistant"):
        with st.spinner("🤔 Agent 思考中..."):
            from agent.agent_graph import run_agent

            response = run_agent(user_input=prompt, use_rag=use_rag)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# ============================================================
# 底部提示
# ============================================================
st.divider()
st.caption(
    "💡 试试：'帮我算一下 sqrt(2) * pi' | '今天几号？' | "
    "'搜索最新的 AI Agent 新闻' | 上传文档后提问"
)
