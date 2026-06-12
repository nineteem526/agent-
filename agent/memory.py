"""
agent/memory.py — 对话记忆管理
===============================
Agent 的"海马体"——让 Agent 能记住之前聊过什么。

两种记忆：
  短期记忆：当前会话的对话历史（存在内存里，LangGraph 的 Checkpointer 自动管理）
  长期记忆：跨会话持久化（存到 ChromaDB，下次打开还能记得）
"""

from typing import List, Dict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class ConversationMemory:
    """
    短期对话记忆管理器。
    本质上就是一个消息列表 + 长度限制（防止超出上下文窗口）。
    """

    def __init__(self, max_turns: int = 20):
        """
        max_turns: 最多保留 20 轮对话（用户+AI 各算一轮）
        超过后自动丢弃最早的对话——这就是"滑动窗口"策略
        """
        self.max_turns = max_turns
        self._messages: List[BaseMessage] = []

    def add_user_message(self, content: str):
        self._messages.append(HumanMessage(content=content))

    def add_ai_message(self, content: str):
        self._messages.append(AIMessage(content=content))

    def get_messages(self) -> List[BaseMessage]:
        """返回当前记忆中的所有消息"""
        return self._messages

    def get_recent(self, n: int = 10) -> List[BaseMessage]:
        """获取最近 n 条消息"""
        return self._messages[-n:]

    def get_history_text(self) -> str:
        """格式化为纯文本（给 Prompt 用）"""
        if not self._messages:
            return "（这是对话的开始）"

        lines = []
        for msg in self._messages[-self.max_turns :]:
            role = "👤 用户" if isinstance(msg, HumanMessage) else "🤖 Agent"
            lines.append(f"{role}：{msg.content}")
        return "\n".join(lines)

    def clear(self):
        self._messages = []

    def __len__(self):
        return len(self._messages)


# ============================================================
# 全局单例
# ============================================================
_memory_instance = None


def get_memory() -> ConversationMemory:
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = ConversationMemory()
    return _memory_instance
