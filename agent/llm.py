"""
agent/llm.py — LLM 调用层
================================
封装 DeepSeek API，提供统一的大模型调用接口。
这里用的是 LangChain 的 ChatOpenAI，因为 DeepSeek 兼容 OpenAI 格式。
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# ============================================================
# 单例模式：全局共用一个 LLM 实例，避免重复初始化
# ============================================================
_llm_instance = None


def get_llm(temperature: float = 0.7) -> ChatOpenAI:
    """
    返回一个配置好的 DeepSeek LLM 实例。

    为什么包一层函数而不是直接暴露变量？
    → 不同场景可能需要不同的 temperature（创作高一点，推理低一点）
    """
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatOpenAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            temperature=temperature,
        )
    else:
        _llm_instance.temperature = temperature
    return _llm_instance
