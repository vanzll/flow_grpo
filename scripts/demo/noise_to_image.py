"""从缓存中选一个 noise，通过 SD3.5-Medium 生成图片。

Usage:
    # 生成 epoch 0、index 0 的 noise 对应的图片
    python scripts/demo/noise_to_image.py --epoch 0 --index 0

    # 生成 reward 最高的那个 noise 对应的图片
    python scripts/demo/noise_to_image.py --best

    # 指定 prompt（默认无 prompt）
    python scripts/demo/noise_to_image.py --epoch 0 --index 0 --prompt "a cat sitting on a chair"

    # 指定输出路径
    python scripts/demo/noise_to_image.py --best --output best_image.png

    # 指定缓存目录
    python scripts/demo/noise_to_image.py --best --cache_dir cache/prior_shaping_particle
"""

import argparse
import glob
import os
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusion3Pipeline


def load_noise_from_cache(cache_dir: str, epoch: int, index: int):
    path = os.path.join(cache_dir, f"epoch_{epoch:06d}.npz")
    with np.load(path) as data:
        noise = torch.from_numpy(data["noises"][index].astype(np.float32))
        reward = float(data["rewards"][index])
    return noise, reward


def find_best_noise(cache_dir: str):
    files = sorted(glob.glob(os.path.join(cache_dir, "epoch_*.npz")))
    best_reward = -float("inf")
    best_noise = None
    best_loc = None

    for f in files:
        with np.load(f) as data:
            rewards = data["rewards"]
            idx = rewards.argmax()
            if rewards[idx] > best_reward:
                best_reward = float(rewards[idx])
                best_noise = torch.from_numpy(data["noises"][idx].astype(np.float32))
                epoch_num = int(os.path.basename(f).split("_")[1].split(".")[0])
                best_loc = (epoch_num, int(idx))

    return best_noise, best_reward, best_loc


def generate_image(noise, prompt, model_id, num_steps, guidance_scale, resolution, device):
    print(f"加载模型 {model_id} ...")
    pipeline = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipeline.to(device)
    pipeline.safety_checker = None

    # 准备 noise: (C, H, W) -> (1, C, H, W)
    latents = noise.unsqueeze(0).to(device, dtype=torch.float16)

    print(f"生成图片 (steps={num_steps}, cfg={guidance_scale}) ...")
    with torch.autocast("cuda"):
        with torch.no_grad():
            if prompt:
                result = pipeline(
                    prompt=prompt,
                    latents=latents,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    output_type="pt",
                    height=resolution,
                    width=resolution,
                )
            else:
                result = pipeline(
                    prompt="",
                    latents=latents,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    output_type="pt",
                    height=resolution,
                    width=resolution,
                )

    image_tensor = result.images[0]  # (C, H, W) in [0, 1]
    image_pil = Image.fromarray(
        (image_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    )
    return image_pil, image_tensor


def main():
    parser = argparse.ArgumentParser(description="从缓存 noise 生成图片")
    parser.add_argument("--cache_dir", type=str, default="cache/prior_shaping")
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--best", action="store_true", help="选择 reward 最高的 noise")
    parser.add_argument("--prompt", type=str, default="", help="文本 prompt")
    parser.add_argument("--output", type=str, default="output.png", help="输出图片路径")
    parser.add_argument("--output_tensor", type=str, default=None,
                        help="输出张量路径 (.pt)，不指定则不保存")
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # 选择 noise
    if args.best:
        noise, reward, (epoch, index) = find_best_noise(args.cache_dir)
        print(f"最高 reward: {reward:.4f} (epoch={epoch}, index={index})")
    elif args.epoch is not None and args.index is not None:
        noise, reward = load_noise_from_cache(args.cache_dir, args.epoch, args.index)
        print(f"Epoch {args.epoch}, Index {args.index}, Reward: {reward:.4f}")
    else:
        parser.error("请指定 --best 或者 --epoch + --index")

    print(f"Noise shape: {noise.shape}, norm: {noise.norm():.2f}")

    # 生成图片
    image_pil, image_tensor = generate_image(
        noise, args.prompt, args.model, args.steps,
        args.guidance_scale, args.resolution, args.device,
    )

    # 保存
    image_pil.save(args.output)
    print(f"图片已保存: {args.output}")

    if args.output_tensor:
        torch.save(image_tensor.cpu(), args.output_tensor)
        print(f"张量已保存: {args.output_tensor}")


if __name__ == "__main__":
    main()
