"""交互式图片生成，支持 SD3 和 FLUX 系列模型。

Usage:
    # 交互模式（启动后选模型、输 prompt）
    python scripts/demo/generate.py

    # 指定模型直接进入
    python scripts/demo/generate.py --model sd3.5-medium

    # 带参数
    python scripts/demo/generate.py --model flux-dev --device cuda:1 --steps 28
"""

import argparse
import os
import torch
from diffusers import StableDiffusion3Pipeline, FluxPipeline


MODEL_PRESETS = {
    "sd3.5-medium": {
        "repo": "stabilityai/stable-diffusion-3.5-medium",
        "pipeline_cls": StableDiffusion3Pipeline,
        "default_steps": 40,
        "default_cfg": 4.5,
    },
    "sd3.5-large": {
        "repo": "stabilityai/stable-diffusion-3.5-large",
        "pipeline_cls": StableDiffusion3Pipeline,
        "default_steps": 40,
        "default_cfg": 4.5,
    },
    "flux-dev": {
        "repo": "black-forest-labs/FLUX.1-dev",
        "pipeline_cls": FluxPipeline,
        "default_steps": 28,
        "default_cfg": 3.5,
    },
    "flux-schnell": {
        "repo": "black-forest-labs/FLUX.1-schnell",
        "pipeline_cls": FluxPipeline,
        "default_steps": 4,
        "default_cfg": 0.0,
    },
}


def select_model_interactive():
    print("\n可用模型：")
    names = list(MODEL_PRESETS.keys())
    for i, name in enumerate(names):
        info = MODEL_PRESETS[name]
        print(f"  [{i+1}] {name:20s}  ({info['repo']})")
    print()
    while True:
        choice = input("选择模型 (输入编号或名称): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            return names[int(choice) - 1]
        if choice in MODEL_PRESETS:
            return choice
        print(f"  无效选择，请输入 1-{len(names)} 或模型名称")


def load_pipeline(model_name, device, dtype=torch.float16):
    preset = MODEL_PRESETS[model_name]
    print(f"\n加载 {model_name} ({preset['repo']}) ...")
    pipeline = preset["pipeline_cls"].from_pretrained(preset["repo"], torch_dtype=dtype)
    pipeline.to(device)
    if hasattr(pipeline, "safety_checker"):
        pipeline.safety_checker = None
    return pipeline, preset


def generate_one(pipeline, prompt, negative_prompt, steps, cfg, resolution, device, seed=None):
    if seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()
    generator = torch.Generator(device=device).manual_seed(seed)

    kwargs = dict(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=cfg,
        height=resolution,
        width=resolution,
        generator=generator,
    )
    # SD3 支持 negative_prompt，FLUX 不支持
    if negative_prompt and isinstance(pipeline, StableDiffusion3Pipeline):
        kwargs["negative_prompt"] = negative_prompt

    with torch.autocast("cuda"):
        with torch.no_grad():
            image = pipeline(**kwargs).images[0]

    return image, seed


def main():
    parser = argparse.ArgumentParser(description="交互式图片生成")
    parser.add_argument("--model", type=str, default=None,
                        help=f"模型名称: {', '.join(MODEL_PRESETS.keys())}")
    parser.add_argument("--prompt", type=str, default=None, help="直接指定 prompt（非交互）")
    parser.add_argument("--negative_prompt", type=str, default="", help="负面 prompt")
    parser.add_argument("--num", type=int, default=1, help="每个 prompt 生成几张")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--steps", type=int, default=None, help="去噪步数（不指定则用模型默认值）")
    parser.add_argument("--cfg", type=float, default=None, help="CFG scale（不指定则用模型默认值）")
    parser.add_argument("--resolution", type=int, default=512, help="图片分辨率")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="outputs/generate")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 选模型
    if args.model and args.model in MODEL_PRESETS:
        model_name = args.model
    else:
        model_name = select_model_interactive()

    pipeline, preset = load_pipeline(model_name, args.device)
    steps = args.steps or preset["default_steps"]
    cfg = args.cfg if args.cfg is not None else preset["default_cfg"]

    print(f"\n当前设置: model={model_name}, steps={steps}, cfg={cfg}, resolution={args.resolution}")
    print(f"输出目录: {args.output_dir}/")

    # 如果指定了 prompt，直接生成
    if args.prompt:
        for i in range(args.num):
            seed = (args.seed + i) if args.seed is not None else None
            image, seed = generate_one(
                pipeline, args.prompt, args.negative_prompt,
                steps, cfg, args.resolution, args.device, seed,
            )
            safe = "".join(c if c.isalnum() or c in " _-" else "" for c in args.prompt)[:50].strip()
            path = os.path.join(args.output_dir, f"{safe}_seed{seed}.png")
            image.save(path)
            print(f"  seed={seed} -> {path}")
        print()

    # 交互模式
    print("输入 prompt 生成图片。特殊命令：")
    print("  /model   - 切换模型")
    print("  /steps N - 修改步数")
    print("  /cfg N   - 修改 CFG")
    print("  /res N   - 修改分辨率")
    print("  /num N   - 修改每次生成数量")
    print("  q        - 退出\n")

    num = args.num
    while True:
        try:
            text = input("prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            break

        if not text or text.lower() == "q":
            break

        # 特殊命令
        if text.startswith("/model"):
            model_name = select_model_interactive()
            pipeline, preset = load_pipeline(model_name, args.device)
            steps = preset["default_steps"]
            cfg = preset["default_cfg"]
            print(f"已切换: model={model_name}, steps={steps}, cfg={cfg}")
            continue
        if text.startswith("/steps"):
            steps = int(text.split()[1])
            print(f"  steps={steps}")
            continue
        if text.startswith("/cfg"):
            cfg = float(text.split()[1])
            print(f"  cfg={cfg}")
            continue
        if text.startswith("/res"):
            args.resolution = int(text.split()[1])
            print(f"  resolution={args.resolution}")
            continue
        if text.startswith("/num"):
            num = int(text.split()[1])
            print(f"  num={num}")
            continue

        # 生成图片
        for i in range(num):
            seed = (args.seed + i) if args.seed is not None else None
            image, seed = generate_one(
                pipeline, text, args.negative_prompt,
                steps, cfg, args.resolution, args.device, seed,
            )
            safe = "".join(c if c.isalnum() or c in " _-" else "" for c in text)[:50].strip()
            path = os.path.join(args.output_dir, f"{safe}_seed{seed}.png")
            image.save(path)
            print(f"  seed={seed} -> {path}")


if __name__ == "__main__":
    main()
