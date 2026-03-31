import os
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video


def generate_video_from_text(
    prompt: str,
    model_path: str,
    output_path: str = "outputs/test_cogvideox.mp4",
    num_inference_steps: int = 12,
    guidance_scale: float = 6.0,
    num_frames: int = 17,
    fps: int = 8,
    height: int = 256,
    width: int = 384,
    seed: int = 42,
):
    """
    用本地 CogVideoX 模型进行文生视频
    先用保守参数，目标是稳定跑通
    """

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路径不存在: {model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("=" * 60)
    print("开始加载本地 CogVideoX 模型")
    print(f"模型路径: {os.path.abspath(model_path)}")
    print(f"设备: {device}")
    print(f"数据类型: {dtype}")
    print("=" * 60)

    pipe = CogVideoXPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype
    )

    if device == "cuda":
        pipe = pipe.to("cuda")
        # 显存优化
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    else:
        pipe = pipe.to("cpu")

    generator = torch.Generator(device=device).manual_seed(seed)

    print("开始生成视频...")
    print(f"Prompt: {prompt}")

    result = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_frames=num_frames,
        height=height,
        width=width,
        generator=generator,
    )

    frames = result.frames[0]
    export_to_video(frames, output_path, fps=fps)

    print("=" * 60)
    print(f"视频生成完成: {output_path}")
    print("=" * 60)
    return output_path


if __name__ == "__main__":
    test_prompt = (
        "A cinematic fantasy scene, a young traveler walking through a glowing forest at night, "
        "blue mist, fireflies, realistic lighting, dynamic camera movement"
    )

    generate_video_from_text(
        prompt=test_prompt,
        model_path=r"./p_au/CogVideoX-2b",
        output_path=r"outputs/test_cogvideox.mp4",
        num_inference_steps=12,
        guidance_scale=6.0,
        num_frames=17,
        fps=8,
        height=256,
        width=384,
        seed=42,
    )