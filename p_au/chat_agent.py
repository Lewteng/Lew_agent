# p_au/chat_agent.py
from openai import OpenAI
from p_au.config import (
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    CHAT_MODEL,
    SYSTEM_PROMPT_ZH
)


class ChatAgent:
    def __init__(self):
        if not DASHSCOPE_API_KEY:
            raise ValueError("DASHSCOPE_API_KEY 未配置，请先在 .env 中填写。")

        self.client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL
        )

        self.history = [
            {"role": "system", "content": SYSTEM_PROMPT_ZH}
        ]

    def add_context(self, story: str = "", image_path: str = ""):
        """
        把最近一次图片生成故事的结果注入到聊天上下文里。
        避免重复注入完全相同的内容。
        """
        context_parts = []

        if image_path:
            context_parts.append(f"最近一次用户使用的图片路径：{image_path}")

        if story:
            context_parts.append(
                "最近一次根据图片生成的故事如下：\n"
                f"{story}"
            )

        if not context_parts:
            return

        context_text = "\n\n".join(context_parts)

        # 简单防重复：如果最近一条 system/context 一样，就不再加
        if self.history and self.history[-1]["role"] == "system" and self.history[-1]["content"] == context_text:
            return

        self.history.append({
            "role": "system",
            "content": context_text
        })

    def chat(self, user_text: str) -> str:
        user_text = user_text.strip()
        if not user_text:
            return "你还没有输入内容。"

        # 固定回复：项目真实能力说明
        if "你有哪些功能" in user_text or "你能做什么" in user_text:
            reply = (
                "我目前是 Lew_agent 项目中的智能助手，当前版本主要支持：\n"
                "1. 多轮对话；\n"
                "2. 根据图片生成故事；\n"
                "3. 将故事文本转成语音；\n"
                "4. 在聊天中结合最近一次生成的图片故事继续交流。\n"
                "视频生成功能目前还没有作为稳定版本接入。"
            )
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": reply})
            return reply

        if "你是什么模型" in user_text or "你是谁" in user_text:
            reply = (
                "我是 Lew_agent 项目中的智能助手。"
                "当前由接入的大模型能力提供对话与图像理解支持，"
                "主要用于多轮对话、图片生成故事和故事转语音。"
            )
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": reply})
            return reply

        self.history.append({"role": "user", "content": user_text})

        stream = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=self.history,
            stream=True
        )

        chunks = []

        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)

            if content:
                if isinstance(content, str):
                    chunks.append(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_part = item.get("text", "")
                            chunks.append(text_part)

        reply = "".join(chunks).strip()
        if not reply:
            reply = "模型没有返回有效内容。"

        self.history.append({"role": "assistant", "content": reply})
        return reply