# p_au/tts_generator.py
import asyncio
import edge_tts


VOICE_PRESETS = {
    "default_female": {
        "voice": "zh-CN-XiaoxiaoNeural",
        "rate": "+0%",
        "pitch": "+0Hz"
    },
    "male_gentle": {
        "voice": "zh-CN-YunxiNeural",
        "rate": "-10%",
        "pitch": "-10Hz"
    },
    "male_story": {
        "voice": "zh-CN-YunxiNeural",
        "rate": "-20%",
        "pitch": "-20Hz"
    },
    "fast_talk": {
        "voice": "zh-CN-YunxiNeural",
        "rate": "+20%",
        "pitch": "+0Hz"
    }
}


async def _tts_async(text, output_path, voice, rate, pitch):
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
        pitch=pitch
    )
    await communicate.save(output_path)


def text_to_speech(text, output_path, preset="male_story"):
    config = VOICE_PRESETS.get(preset, VOICE_PRESETS["default_female"])

    asyncio.run(_tts_async(
        text,
        output_path,
        config["voice"],
        config["rate"],
        config["pitch"]
    ))