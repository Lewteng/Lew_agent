# p_au/vision_story.py
import base64
from pathlib import Path
from openai import OpenAI

from p_au.config import (
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    VISION_MODEL,
    DEFAULT_STORY_PROMPT_ZH,
    SYSTEM_PROMPT_ZH,
)


def image_to_base64(image_path: str) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"图片不存在：{image_path}")

    suffix = path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    mime_type = mime_map.get(suffix, "image/jpeg")

    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def generate_story_from_image(image_path: str, user_prompt: str = "") -> str:
    if not DASHSCOPE_API_KEY:
        raise ValueError("DASHSCOPE_API_KEY 未配置，请先在 .env 中填写。")

    client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL
    )

    prompt = user_prompt.strip() if user_prompt.strip() else DEFAULT_STORY_PROMPT_ZH
    image_data_url = image_to_base64(image_path)

    stream = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_ZH},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url}
                    }
                ]
            }
        ],
        stream=True
    )

    chunks = []
    print("\n[模型生成中...]\n")

    for chunk in stream:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)

        if content:
            # 有些SDK直接给字符串
            if isinstance(content, str):
                print(content, end="", flush=True)
                chunks.append(content)

            # 有些兼容接口可能给分段结构
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_part = item.get("text", "")
                        print(text_part, end="", flush=True)
                        chunks.append(text_part)

    print("\n")
    story = "".join(chunks).strip()

    if not story:
        raise RuntimeError("模型未返回有效故事文本。")

    return story