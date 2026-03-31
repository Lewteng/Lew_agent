# p_au/app.py
from pathlib import Path
from datetime import datetime
import os
import platform
import subprocess

from p_au.config import DEFAULT_IMAGE_PATH
from p_au.vision_story import generate_story_from_image
from p_au.chat_agent import ChatAgent
from p_au.tts_generator import text_to_speech


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

last_story = ""
last_image_path = ""

# 全局持久聊天代理：不会因为退出聊天菜单而丢失上下文
agent = ChatAgent()


def auto_play_audio(audio_path: Path):
    """
    尝试自动播放音频文件
    Windows / macOS / Linux 做基础兼容
    """
    try:
        system_name = platform.system()

        if system_name == "Windows":
            os.startfile(str(audio_path))
        elif system_name == "Darwin":  # macOS
            subprocess.run(["open", str(audio_path)], check=False)
        else:  # Linux
            subprocess.run(["xdg-open", str(audio_path)], check=False)

    except Exception as e:
        print(f"[提示] 自动播放失败，请手动打开音频文件：{audio_path}")
        print(f"[自动播放失败原因] {e}")


def run_image_story_tts():
    global last_story, last_image_path, agent

    try:
        print("\n=== 图片 -> 故事 -> 语音 模式 ===")
        image_path = input(f"请输入图片路径（直接回车默认 {DEFAULT_IMAGE_PATH}）：").strip()
        if not image_path:
            image_path = DEFAULT_IMAGE_PATH

        user_prompt = input("请输入故事提示词（直接回车使用默认）：").strip()

        print("\n正在根据图片生成故事...")
        story = generate_story_from_image(image_path, user_prompt)

        # 保存最近一次状态
        last_story = story
        last_image_path = image_path

        # 同步注入聊天上下文
        agent.add_context(story=last_story, image_path=last_image_path)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        story_path = OUTPUT_DIR / f"story_{timestamp}.txt"
        audio_path = OUTPUT_DIR / f"speech_{timestamp}.mp3"

        story_path.write_text(story, encoding="utf-8")

        print("\n=== 生成的故事 ===")
        print(story)

        print("\n请选择语音风格：")
        print("1. 默认女声")
        print("2. 温和男声（推荐）")
        print("3. 讲故事男声（推荐🔥）")
        print("4. 快速语音")

        voice_choice = input("选择：").strip()

        voice_map = {
            "1": "default_female",
            "2": "male_gentle",
            "3": "male_story",
            "4": "fast_talk"
        }

        preset = voice_map.get(voice_choice, "male_story")

        print("\n正在生成语音...")
        text_to_speech(story, str(audio_path), preset)

        print(f"\n故事已保存：{story_path}")
        print(f"语音已保存：{audio_path}")
        print("\n本次结果已保存到 outputs 文件夹。")

        print("\n正在自动播放语音...")
        auto_play_audio(audio_path)

    except Exception as e:
        print(f"\n[错误] 图片生成故事或语音失败：{e}\n")


def run_chat():
    global agent, last_story, last_image_path

    try:
        print("\n=== 多轮对话模式 ===")
        print("输入 exit 退出。\n")

        # 如果之前生成过故事，确保聊天时能拿到这段上下文
        if last_story or last_image_path:
            agent.add_context(story=last_story, image_path=last_image_path)

        while True:
            user_text = input("你：").strip()
            if user_text.lower() in ["exit", "quit", "q"]:
                print("已退出聊天。")
                break

            reply = agent.chat(user_text)
            print(f"\nLew_agent：{reply}\n")

    except Exception as e:
        print(f"\n[错误] 聊天模式运行失败：{e}\n")


def main():
    while True:
        print("\n========== Lew_agent ==========")
        print("1. 图片生成故事并语音播报")
        print("2. 多轮文字对话")
        print("0. 退出")

        choice = input("请选择功能：").strip()

        if choice == "1":
            run_image_story_tts()
        elif choice == "2":
            run_chat()
        elif choice == "0":
            print("程序结束。")
            break
        else:
            print("输入无效，请重新选择。")


if __name__ == "__main__":
    main()