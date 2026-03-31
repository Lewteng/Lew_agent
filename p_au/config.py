# p_au/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# ===== 阿里云 / DashScope OpenAI兼容接口 =====
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()

# 这里你按你实际地域改
# 国际站常见是 compatible-mode/v1
# 中国内地如果你用的是百炼/Model Studio，对应 endpoint 以你控制台为准
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
).strip()

# 你现在可用的模型
VISION_MODEL = os.getenv("VISION_MODEL", "qvq-max-2025-03-25").strip()
CHAT_MODEL = os.getenv("CHAT_MODEL", "qvq-max-2025-03-25").strip()

DEFAULT_IMAGE_PATH = os.getenv(
    "DEFAULT_IMAGE_PATH",
    r"D:\PyCharm2023.1.4\pythonproject\Lew_agent\p_au\assets\test.jpg"
)

DEFAULT_STORY_PROMPT_ZH = """
请基于这张图片创作一个完整中文故事。
要求：
1. 有明确的场景、人物、动作和情绪；
2. 语言自然，有画面感；
3. 篇幅控制在 300~500 字；
4. 不要分点，直接输出故事正文。
""".strip()

SYSTEM_PROMPT_ZH = """
你是 Lew_agent 项目中的智能助手，名字就叫 Lew_agent。
请始终以 Lew_agent 助手的身份与用户交流。

要求：
1. 默认使用中文回答，除非用户明确要求英文；
2. 回答自然、简洁、清晰；
3. 不要主动声称自己是某个底层模型；
4. 当用户问功能时，优先围绕 Lew_agent 当前实际功能回答；
5. 当前稳定支持的功能主要包括：
   - 多轮对话
   - 根据图片生成故事
   - 将故事转成语音
6. 不要夸大未实现或未稳定接入的能力；
7. 视频功能当前不是稳定主线，如被问到，应如实说明。
""".strip()