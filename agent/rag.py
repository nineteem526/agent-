"""
agent/rag.py — RAG 检索增强生成系统
=====================================
完整 RAG 流程：文档加载 → 文本分块 → 向量化 → 存库 → 检索

核心概念复习：
  Embedding（向量化）：把文字变成数学向量，语义相近的向量也相近
  Chunking（分块）：长文档切成小段，每段单独向量化
  ChromaDB：本地向量数据库，"找最相似的 Top-K 段落"
"""

import os
from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# ============================================================
# Embedding 模型：用本地模型，零 API 成本
# ============================================================
# text2vec-base-chinese：专为中文优化的轻量向量模型，本地运行。
# 首次使用会自动下载模型（约 400MB），之后完全离线。
# 国内用户用 hf-mirror.com 镜像下载。

# ============================================================
# 设置 HuggingFace 镜像（国内访问更快）
# 注意：必须在 import sentence_transformers 之前设置
# ============================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

_embeddings = None  # 延迟初始化
_rag_available = True  # 标记 RAG 是否可用


def get_embeddings():
    """
    获取 embedding 模型实例。
    如果下载失败（网络问题），返回 None 并禁用 RAG，但不影响 Agent 其他功能。
    """
    global _embeddings, _rag_available

    if not _rag_available:
        return None

    if _embeddings is None:
        try:
            # 禁用 huggingface_hub 的自动重试（每次 8s × 5 次 = 40s 卡顿）
            os.environ.setdefault("HF_HUB_DISABLE_RETRY", "1")
            os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "10")

            from langchain_community.embeddings import HuggingFaceEmbeddings

            print("📥 正在加载 embedding 模型...")
            _embeddings = HuggingFaceEmbeddings(
                model_name="shibing624/text2vec-base-chinese",
            )
            print("✅ Embedding 模型加载完成，RAG 已就绪")
        except Exception:
            _rag_available = False
            print("⚠️  Embedding 模型下载失败（网络不通），RAG 暂不可用")
            print("💡 Agent 工具调用（计算/搜索/时间）完全正常，继续使用")
            return None
    return _embeddings


def is_rag_available() -> bool:
    """检查 RAG 功能是否可用"""
    return _rag_available


# ============================================================
# 文档加载
# ============================================================
def load_document(file_path: str) -> List[Document]:
    """
    加载 PDF 或 TXT 文件，返回 LangChain Document 列表。
    每个 Document 包含：page_content（文本） + metadata（来源信息）
    """
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt") or file_path.endswith(".md"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"不支持的文件格式：{file_path}。仅支持 PDF / TXT / MD。")

    docs = loader.load()
    print(f"📄 已加载文档：{file_path}，共 {len(docs)} 页/段")
    return docs


# ============================================================
# 文本分块（Chunking）
# ============================================================
def split_documents(
    docs: List[Document],
    chunk_size: int = 500,  # 每块 500 字符
    chunk_overlap: int = 100,  # 相邻块重叠 100 字符（防止一句话被切断）
) -> List[Document]:
    """
    把长文档切成小块。
    为什么分块？
    → LLM 上下文有限，一次只能读几段。块太大塞不进，块太小丢失上下文。
    → chunk_overlap 保证跨块的信息不丢失。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"✂️  已分块：{len(docs)} 个文档 → {len(chunks)} 个文本块")
    return chunks


# ============================================================
# 向量存储
# ============================================================
def create_vectorstore(
    chunks: List[Document],
    persist_dir: str = "./data/chroma_db",
):
    """
    将文本块向量化并存入 ChromaDB。
    persist_dir：持久化目录，下次启动不用重新向量化。
    返回：Chroma 向量库实例，或 None（如果 embedding 模型不可用）
    """
    emb = get_embeddings()
    if emb is None:
        print("❌ RAG 不可用：embedding 模型未加载，无法创建向量库")
        return None
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=emb,
        persist_directory=persist_dir,
    )
    print(f"💾 已存入向量数据库：{len(chunks)} 条记录 → {persist_dir}")
    return vectorstore


def load_vectorstore(persist_dir: str = "./data/chroma_db"):
    """加载已有的向量数据库。返回 Chroma 实例或 None。"""
    emb = get_embeddings()
    if emb is None:
        return None
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=emb,
    )


# ============================================================
# 检索（Retrieval）
# ============================================================
def retrieve(
    query: str,
    vectorstore: Chroma,
    top_k: int = 3,
) -> List[Tuple[Document, float]]:
    """
    根据用户问题，从向量库中检索最相关的 Top-K 文本块。
    返回：(文档块, 相似度分数) 的列表。
    """
    results = vectorstore.similarity_search_with_relevance_scores(query, k=top_k)
    return results


def format_retrieved_context(results: List[Tuple[Document, float]]) -> str:
    """
    把检索结果格式化为一段提示词，喂给 LLM。
    每个结果标注来源页码和相似度分数。
    """
    if not results:
        return "（未找到相关文档内容）"

    parts = []
    for i, (doc, score) in enumerate(results, 1):
        page = doc.metadata.get("page", "未知")
        parts.append(f"[来源{i} · 第{page}页 · 相关度{score:.2f}]\n{doc.page_content}")

    return "\n\n---\n\n".join(parts)
