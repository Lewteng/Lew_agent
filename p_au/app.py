import os
import json
import re
import math
import wave
import asyncio
import numpy as np
import torch
import edge_tts
import json
from PIL import ImageDraw, ImageFont
import argparse
from diffusers import I2VGenXLPipeline
from diffusers.utils import export_to_video

from PIL import Image, ImageDraw, ImageFont
from moviepy import ImageClip, AudioFileClip, CompositeVideoClip, TextClip
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VLM_MODEL_PATH = os.path.join(BASE_DIR, "Qwen2.5-VL-3B-Instruct")
TTS_MODEL_EN = os.path.join(BASE_DIR, "mms-tts-eng")

IMAGE_PATH = os.path.join(BASE_DIR, "assets", "test.jpg")
AUDIO_PATH = os.path.join(BASE_DIR, "outputs", "story.wav")
VIDEO_PATH = os.path.join(BASE_DIR, "outputs", "story.mp4")


class ImageStoryTeller:
    def __init__(self):
        self.use_cuda = torch.cuda.is_available()
        self.device = "cuda" if self.use_cuda else "cpu"

        print(f"[INFO] Device: {self.device.upper()}")

        # ====== VLM ======
        print("[INFO] Loading VLM model...")
        self.processor = AutoProcessor.from_pretrained(VLM_MODEL_PATH, local_files_only=True)
        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            VLM_MODEL_PATH,
            torch_dtype=torch.float16 if self.use_cuda else torch.float32,
            local_files_only=True,
        ).to(self.device)

        # ====== TTS（英文） ======
        print("[INFO] Loading TTS model...")
        self.tts_en = pipeline(
            task="text-to-speech",
            model=TTS_MODEL_EN,
            device=0 if self.use_cuda else -1,
        )

        # ====== Video（图文生视频） ======
        print("[INFO] Loading video generation model...")
        self.video_model_id = "ali-vilab/i2vgen-xl"

        video_dtype = torch.float16 if self.use_cuda else torch.float32
        self.video_pipe = I2VGenXLPipeline.from_pretrained(
            self.video_model_id,
            torch_dtype=video_dtype,
        )

        if self.use_cuda:
            self.video_pipe.enable_model_cpu_offload()
        else:
            self.video_pipe.to("cpu")



    # ========= 1️⃣ 生成故事（带提示词） =========
    def generate_story(self, image_path: str, prompt: str):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            text=[prompt_text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            generated_ids = self.vlm.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated_trimmed = generated_ids[:, prompt_len:]

        story = self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
        )[0]

        return story.strip()

    # ========= 2️⃣ 语音生成 =========
    async def _edge_tts_save(self, text: str, output_path: str, voice: str):
        communicate = edge_tts.Communicate(text=text, voice=voice)
        await communicate.save(output_path)

    def text_to_speech_zh(self, text: str, output_path: str, voice: str = "zh-CN-XiaoxiaoNeural"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if os.path.exists(output_path):
            os.remove(output_path)
        asyncio.run(self._edge_tts_save(text, output_path, voice))

    def text_to_speech_en(self, text: str, output_path: str):
        audio_out = self.tts_en(text)
        audio = np.asarray(audio_out["audio"], dtype=np.float32)
        sr = int(audio_out["sampling_rate"])

        if audio.ndim > 1:
            audio = np.squeeze(audio)

        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if os.path.exists(output_path):
            os.remove(output_path)

        import wave
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(audio_int16.tobytes())

    def text_to_speech(self, text: str, language: str, output_path: str):
        if language == "zh":
            self.text_to_speech_zh(text, output_path)
        else:
            self.text_to_speech_en(text, output_path)

    def split_sentences(self, text: str):
        parts = re.split(r'(?<=[。！？!?\.])\s*', text.strip())
        parts = [p.strip() for p in parts if p.strip()]
        return parts if parts else [text.strip()]

    def story_to_video_prompt(self, story: str, language: str = "zh"):
        """
        把长故事压缩成适合视频模型的一条 prompt
        """
        if language == "zh":
            compress_prompt = f"""
请将下面的故事压缩成一条适合图文生视频模型使用的中文视频提示词。

要求：
1. 只输出一段提示词，不要解释。
2. 包含：主体、场景、动作、镜头感、氛围、光线、风格。
3. 语言精炼，适合生成 3~5 秒短视频。
4. 保持和原故事一致，不要编造太多新情节。

故事：
{story}
"""
        else:
            compress_prompt = f"""
Please compress the following story into ONE concise video-generation prompt.

Requirements:
1. Output only one prompt, no explanation.
2. Include subject, scene, action, cinematic camera feeling, atmosphere, lighting, and style.
3. Keep it suitable for a 3-5 second short video.
4. Stay faithful to the story.

Story:
{story}
"""

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": compress_prompt}],
            }
        ]

        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[prompt_text],
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            generated_ids = self.vlm.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated_trimmed = generated_ids[:, prompt_len:]

        result = self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return result

    def generate_video_direct(
            self,
            image_path: str,
            story: str,
            output_path: str,
            language: str = "zh",
            num_frames: int = 24,
            num_inference_steps: int = 30,
            guidance_scale: float = 8.5,
            fps: int = 8,
    ):
        """
        使用 Hugging Face I2VGen-XL：
        图片 + 故事压缩后的 prompt -> 视频
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        video_prompt = self.story_to_video_prompt(story, language)
        print("\n[VIDEO PROMPT]")
        print(video_prompt)

        image = Image.open(image_path).convert("RGB").resize((512, 512))

        generator = torch.manual_seed(42)

        result = self.video_pipe(
            prompt=video_prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_frames=num_frames,
            generator=generator,
        )

        frames = result.frames[0]
        export_to_video(frames, output_path, fps=fps)

        return output_path

    def attach_audio_to_video(self, video_path: str, audio_path: str, output_path: str):
        from moviepy import VideoFileClip

        v = VideoFileClip(video_path)
        a = AudioFileClip(audio_path)

        if a.duration > v.duration:
            a = a.subclipped(0, v.duration)
        else:
            a = a.with_duration(v.duration)

        final = v.with_audio(a)
        final.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
        )

        v.close()
        a.close()
        final.close()



    # ========= 主流程 =========
    def run(self, image_path, prompt, language="zh"):
        import time

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        audio_path = os.path.join(BASE_DIR, "outputs", f"story_{timestamp}.wav")
        video_path = os.path.join(BASE_DIR, "outputs", f"story_{timestamp}.mp4")
        shots_dir = os.path.join(BASE_DIR, "outputs", f"shots_{timestamp}")

        print("\n[STEP 1] Generating story...")
        story = self.generate_story(image_path, prompt)

        print("\n===== STORY =====\n")
        print(story)

        print("\n[STEP 2] Generating speech...")
        self.text_to_speech(story, language, audio_path)

        print("\n[STEP 3] Converting story to storyboard shots...")
        shots = self.story_to_shots(story, num_shots=4)
        for i, s in enumerate(shots, 1):
            print(f"[SHOT {i}] {s}")

        print("\n[STEP 4] Generating shot images...")
        shot_images = self.generate_shot_images(shots, shots_dir)

        print("\n[STEP 5] Creating video from shot images...")
        self.create_video_from_shot_images(
            image_paths=shot_images,
            audio_path=audio_path,
            output_path=video_path,
            subtitle_text=story,
        )

        print("\n✅ 完成！")
        print(f"音频: {audio_path}")
        print(f"视频: {video_path}")
        print(f"分镜图目录: {shots_dir}")


if __name__ == "__main__":
    app = ImageStoryTeller()

    print("\n========== 智能故事生成系统 ==========\n")

    # 1️⃣ 语言选择
    language = input("请选择语言（zh/en，默认 zh）：").strip().lower()
    if language not in ["zh", "en"]:
        language = "zh"

    # 2️⃣ 提示词输入
    custom_prompt = input("请输入故事提示词（直接回车使用默认）：").strip()

    if not custom_prompt:
        if language == "zh":
            custom_prompt = "请根据这张图片写一个温暖、连贯、适合朗读的中文短篇故事，语言自然，有画面感。"
        else:
            custom_prompt = "Please write a warm, coherent, vivid short story based on this image, suitable for narration."

    # 3️⃣ 图片路径
    image_path = input(f"请输入图片路径（直接回车使用默认 {IMAGE_PATH}）：").strip()
    if not image_path:
        image_path = IMAGE_PATH

    # 4️⃣ 输出模式
    print("\n请选择输出模式：")
    print("1 - 只生成音频")
    print("2 - 只生成视频")
    print("3 - 全部生成（默认）")

    mode = input("> ").strip()

    if mode not in ["1", "2", "3"]:
        mode = "3"

    print("\n========== 开始生成 ==========\n")

    # 5️⃣ 生成故事
    story = app.generate_story(image_path, custom_prompt)

    print("\n生成的故事：\n")
    print(story)
    print("\n------------------------------\n")

    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    audio_path = os.path.join(BASE_DIR, "outputs", f"story_{timestamp}.wav")
    video_path = os.path.join(BASE_DIR, "outputs", f"story_{timestamp}.mp4")
    shot_dir = os.path.join(BASE_DIR, "outputs", f"shots_{timestamp}")

    # 6️⃣ 音频分支
    if mode in ["1", "3"]:
        print("[INFO] 生成语音...")
        app.text_to_speech(story, language, audio_path)
        print(f"[DONE] 音频已保存：{audio_path}")

    # 7️⃣ 视频分支
    if mode in ["2", "3"]:
        print("[INFO] 生成分镜...")
        shots = app.story_to_shots(story)

        print("[INFO] 生成分镜图...")
        shot_images = app.generate_shot_images(shots, shot_dir)

        print("[INFO] 合成视频...")
        app.create_video_from_shot_images(
            shot_images,
            audio_path if os.path.exists(audio_path) else None,
            video_path,
            story
        )

        print(f"[DONE] 视频已保存：{video_path}")

    print("\n========== 全部完成 ==========\n")