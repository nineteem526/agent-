"""
api.py — FastAPI REST 接口
===========================
为 Agent 提供 HTTP API，支持程序化调用和前端集成。

运行：uvicorn api:app --reload --host 0.0.0.0 --port 8000
文档：启动后访问 http://localhost:8000/docs (Swagger UI)
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# HuggingFace 镜像（必须在 import agent 之前设置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_RETRY", "1")  # 加速降级，不重试

# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(
    title="AI Agent API",
    description="多工具 ReAct Agent 的 RESTful 接口，支持对话、RAG 文档检索、工具调用",
    version="1.0.0",
)

# CORS：允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 请求/响应模型
# ============================================================
class ChatRequest(BaseModel):
    message: str = Field(..., description="用户输入的问题", min_length=1)
    use_rag: bool = Field(default=True, description="是否启用 RAG 检索")
    thread_id: str = Field(default="default", description="对话线程 ID（不同线程独立记忆）")


class ChatResponse(BaseModel):
    response: str = Field(..., description="Agent 的回答")
    thread_id: str = Field(default="default", description="对话线程 ID")


class HealthResponse(BaseModel):
    status: str
    rag_available: bool
    tools: List[str]


# ============================================================
# 启动时初始化 Agent（只初始化一次）
# ============================================================
_initialized = False

@app.on_event("startup")
def startup():
    """服务启动时只初始化 Agent 图（不触发 embedding 下载）"""
    global _initialized
    if not _initialized:
        from agent.agent_graph import get_agent_app
        get_agent_app()
        _initialized = True
        print("[OK] Agent API ready (RAG lazy-init on first doc upload or query)")


# ============================================================
# API 端点
# ============================================================
@app.get("/health", response_model=HealthResponse)
def health_check():
    """健康检查 + 服务状态"""
    from agent.rag import is_rag_available
    from agent.tools import ALL_TOOLS

    return HealthResponse(
        status="ok",
        rag_available=is_rag_available(),
        tools=[t.name for t in ALL_TOOLS],
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    向 Agent 发送消息，获取回复。

    示例：
      curl -X POST http://localhost:8000/chat \\
        -H "Content-Type: application/json" \\
        -d '{"message": "帮我算 sqrt(16) * 3.14", "use_rag": false}'
    """
    from agent.agent_graph import run_agent

    # 首次 RAG 请求时懒初始化
    if req.use_rag:
        from agent.agent_graph import init_rag
        init_rag()

    try:
        answer = run_agent(
            user_input=req.message,
            use_rag=req.use_rag,
            thread_id=req.thread_id,
        )
        return ChatResponse(response=answer, thread_id=req.thread_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/upload")
def upload_documents(files: List[UploadFile] = File(..., description="PDF/TXT/MD 文件")):
    """
    上传文档到 RAG 知识库。

    示例：
      curl -X POST http://localhost:8000/documents/upload \\
        -F "files=@report.pdf" -F "files=@notes.txt"
    """
    from agent.agent_graph import reindex_documents
    from agent.rag import is_rag_available

    if not is_rag_available():
        raise HTTPException(
            status_code=503,
            detail="RAG 不可用：embedding 模型未下载，无法索引文档",
        )

    allowed_ext = {".pdf", ".txt", ".md"}
    temp_dir = tempfile.mkdtemp()
    file_paths = []

    try:
        for f in files:
            ext = Path(f.filename).suffix.lower()
            if ext not in allowed_ext:
                raise HTTPException(
                    status_code=400,
                    detail=f"不支持的文件类型：{ext}（仅支持 PDF/TXT/MD）",
                )
            path = os.path.join(temp_dir, f.filename)
            with open(path, "wb") as wf:
                wf.write(f.file.read())
            file_paths.append(path)

        chunk_count = reindex_documents(file_paths, "./data/chroma_db")
        return {
            "status": "ok",
            "files_count": len(files),
            "chunk_count": chunk_count,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.delete("/memory/{thread_id}")
def clear_memory(thread_id: str = "default"):
    """清除指定线程的对话记忆"""
    from agent.memory import get_memory

    get_memory().clear()
    return {"status": "ok", "thread_id": thread_id, "message": "记忆已清除"}


# ============================================================
# 直接运行入口
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
