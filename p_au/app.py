import os
import re
import time
import base64
import asyncio
from pathlib import Path

import edge_tts
from dotenv import load_dotenv
from PIL import Image
from openai import OpenAI
from huggingface_hub import InferenceClient
from moviepy import AudioFileClip, VideoFileClip

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

IMAGE_PATH = BASE_DIR / "assets" / "test.jpg"
OUTPUT_DIR = BASE_DIR / "outputs"

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
HF_VLM_MODEL = os.getenv("HF_VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct").strip()
HF_T2V_PROVIDER = os.getenv("HF_T2V_PROVIDER", "fal-ai").strip()
HF_T2V_MODEL = os.getenv("HF_T2V_MODEL", "").strip()  # 可留空

if not HF_TOKEN:
    raise ValueError("未检测到 HF_TOKEN，请先在 .env 或系统环境变量中设置。")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def image_to_data_url(image_path: str) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"图片不存在：{image_path}")

    suffix = path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    mime = mime_map.get(suffix, "application/octet-stream")

    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


class ImageStoryTellerOnline:
    def __init__(self) -> None:
        self.openai_client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=HF_TOKEN,
        )

        self.video_client = InferenceClient(
            provider=HF_T2V_PROVIDER,
            api_key=HF_TOKEN,
        )

        ensure_dir(OUTPUT_DIR)

    # ========= 1️⃣ 在线生成故事（图片 + 提示词） =========
    def generate_story(self, image_path: str, prompt: str) -> str:
        image_data_url = image_to_data_url(image_path)

        response = self.openai_client.responses.create(
            model=HF_VLM_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_data_url},
                    ],
                }
            ],
        )
        story = (response.output_text or "").strip()
        if not story:
            raise RuntimeError("故事生成失败：模型返回为空。")
        return story

    # ========= 2️⃣ 在线压缩视频提示词 =========
    def story_to_video_prompt(self, story: str, language: str = "zh") -> str:
        if language == "zh":
            compress_prompt = f"""
请将下面的故事压缩成一条适合文生视频模型使用的中文视频提示词。

要求：
1. 只输出一段提示词，不要解释。
2. 包含：主体、场景、动作、镜头感、氛围、光线、风格。
3. 语言精炼，适合生成 3~5 秒短视频。
4. 保持和原故事一致，不要编造太多新情节。

故事：
{story}
""".strip()
        else:
            compress_prompt = f"""
Please compress the following story into ONE concise text-to-video prompt.

Requirements:
1. Output only one prompt, no explanation.
2. Include subject, scene, action, cinematic camera feeling, atmosphere, lighting, and style.
3. Keep it suitable for a 3-5 second short video.
4. Stay faithful to the story.

Story:
{story}
""".strip()

        response = self.openai_client.responses.create(
            model=HF_VLM_MODEL,
            input=compress_prompt,
        )
        video_prompt = (response.output_text or "").strip()
        if not video_prompt:
            raise RuntimeError("视频提示词生成失败：模型返回为空。")
        return video_prompt

    # ========= 3️⃣ 在线 TTS（中英文统一 edge_tts） =========
    async def _edge_tts_save(self, text: str, output_path: str, voice: str) -> None:
        communicate = edge_tts.Communicate(text=text, voice=voice)
        await communicate.save(output_path)

    def text_to_speech(self, text: str, language: str, output_path: str) -> None:
        ensure_dir(Path(output_path).parent)

        voice = "zh-CN-XiaoxiaoNeural" if language == "zh" else "en-US-JennyNeural"
        if Path(output_path).exists():
            Path(output_path).unlink()

        asyncio.run(self._edge_tts_save(text, output_path, voice))

    # ========= 4️⃣ 在线文生视频 =========
    def generate_video_direct(
        self,
        story: str,
        output_path: str,
        language: str = "zh",
        num_frames: int = 24,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
    ) -> str:
        ensure_dir(Path(output_path).parent)

        video_prompt = self.story_to_video_prompt(story, language)
        print("\n[VIDEO PROMPT]")
        print(video_prompt)

        kwargs = {
            "prompt": video_prompt,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }

        # 模型可留空：让 provider 走默认推荐/默认配置
        if HF_T2V_MODEL:
            video_bytes = self.video_client.text_to_video(
                video_prompt,
                model=HF_T2V_MODEL,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
        else:
            video_bytes = self.video_client.text_to_video(
                video_prompt,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )

        with open(output_path, "wb") as f:
            f.write(video_bytes)

        return output_path

    # ========= 5️⃣ 给视频挂音频 =========
    def attach_audio_to_video(self, video_path: str, audio_path: str, output_path: str) -> str:
        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)

        if audio.duration > video.duration:
            audio = audio.subclipped(0, video.duration)
        else:
            audio = audio.with_duration(video.duration)

        final = video.with_audio(audio)
        final.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
        )

        video.close()
        audio.close()
        final.close()
        return output_path

    # ========= 6️⃣ 主流程 =========
    def run(self, image_path: str, prompt: str, language: str = "zh", mode: str = "3") -> None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        story_txt_path = OUTPUT_DIR / f"story_{timestamp}.txt"
        audio_path = OUTPUT_DIR / f"story_{timestamp}.mp3"
        raw_video_path = OUTPUT_DIR / f"story_raw_{timestamp}.mp4"
        final_video_path = OUTPUT_DIR / f"story_{timestamp}.mp4"

        print("\n[STEP 1] Generating story...")
        story = self.generate_story(image_path, prompt)

        print("\n===== STORY =====\n")
        print(story)

        story_txt_path.write_text(story, encoding="utf-8")
        print(f"\n[DONE] 故事已保存：{story_txt_path}")

        if mode in ["1", "3"]:
            print("\n[STEP 2] Generating speech...")
            self.text_to_speech(story, language, str(audio_path))
            print(f"[DONE] 音频已保存：{audio_path}")

        if mode in ["2", "3"]:
            print("\n[STEP 3] Generating video...")
            self.generate_video_direct(
                story=story,
                output_path=str(raw_video_path),
                language=language,
            )
            print(f"[DONE] 原始视频已保存：{raw_video_path}")

            if mode == "3" and audio_path.exists():
                print("\n[STEP 4] Attaching audio to video...")
                self.attach_audio_to_video(
                    video_path=str(raw_video_path),
                    audio_path=str(audio_path),
                    output_path=str(final_video_path),
                )
                print(f"[DONE] 成品视频已保存：{final_video_path}")

        print("\n========== 全部完成 ==========\n")


if __name__ == "__main__":
    app = ImageStoryTellerOnline()

    print("\n========== 智能故事生成系统（在线版） ==========\n")

    language = input("请选择语言（zh/en，默认 zh）：").strip().lower()
    if language not in ["zh", "en"]:
        language = "zh"

    custom_prompt = input("请输入故事提示词（直接回车使用默认）：").strip()
    if not custom_prompt:
        if language == "zh":
            custom_prompt = "请根据这张图片写一个温暖、连贯、适合朗读的中文短篇故事，语言自然，有画面感。"
        else:
            custom_prompt = "Please write a warm, coherent, vivid short story based on this image, suitable for narration."

    image_path = input(f"请输入图片路径（直接回车使用默认 {IMAGE_PATH}）：").strip()
    if not image_path:
        image_path = str(IMAGE_PATH)

    print("\n请选择输出模式：")
    print("1 - 只生成音频")
    print("2 - 只生成视频")
    print("3 - 全部生成（默认）")

    mode = input("> ").strip()
    if mode not in ["1", "2", "3"]:
        mode = "3"

    print("\n========== 开始生成 ==========\n")
    app.run(image_path=image_path, prompt=custom_prompt, language=language, mode=mode)