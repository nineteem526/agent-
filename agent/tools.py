"""
agent/tools.py — 工具系统
==========================
Agent 的"手脚"——LLM 能调用的外部函数。
每个工具就是一个函数 + 一段描述（告诉 LLM 什么时候用它）。
"""

import json
import math
from typing import Optional
from langchain_core.tools import tool
from duckduckgo_search import DDGS


# ============================================================
# 工具 1：计算器
# ============================================================
@tool
def calculator(expression: str) -> str:
    """
    执行数学计算。支持 + - * / ** sqrt() sin() cos() log() 等。
    输入：一个数学表达式字符串，如 "sqrt(144) + 3 * 5"
    """
    try:
        # 安全计算：限制可用函数，防止代码注入
        allowed_names = {
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "log": math.log,
            "log10": math.log10,
            "pi": math.pi,
            "e": math.e,
            "abs": abs,
            "round": round,
            "pow": pow,
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"✅ 计算结果：{result}"
    except Exception as e:
        return f"❌ 计算出错：{str(e)}。请检查表达式格式。"


# ============================================================
# 工具 2：网络搜索（免费，不需要 API Key）
# ============================================================
@tool
def web_search(query: str, max_results: int = 3) -> str:
    """
    在互联网上搜索最新信息。当需要实时信息或 LLM 训练数据中没有的知识时使用。
    输入：搜索关键词。
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return f"🔍 未找到关于「{query}」的搜索结果。"

            formatted = []
            for i, r in enumerate(results, 1):
                formatted.append(
                    f"{i}. {r['title']}\n   {r['body'][:200]}...\n   🔗 {r['href']}"
                )
            return "🔍 搜索结果：\n\n" + "\n\n".join(formatted)
    except Exception as e:
        return f"❌ 搜索失败：{str(e)}"


# ============================================================
# 工具 3：日期时间
# ============================================================
@tool
def get_current_time() -> str:
    """
    获取当前日期和时间。当用户问"现在几点""今天几号"时使用。
    """
    from datetime import datetime

    now = datetime.now()
    weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    return f"🕐 当前时间：{now.strftime('%Y年%m月%d日')} {weekdays[now.weekday()]} {now.strftime('%H:%M:%S')}"


# ============================================================
# 工具汇总
# ============================================================
# 所有工具注册在这里。以后加新工具只需写函数 + 加进这个列表。
ALL_TOOLS = [calculator, web_search, get_current_time]
