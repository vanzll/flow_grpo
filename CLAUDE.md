# CLAUDE.md

## Project Overview

Flow-GRPO: Training flow matching models (diffusion models) via online reinforcement learning (GRPO). Research codebase supporting SD3.5-M, FLUX.1-dev, FLUX.1-Kontext, Qwen-Image, Wan2.1, and Bagel-7B.

## Setup

```bash
conda create -n flow_grpo python=3.10.16
pip install -e .
```

Some reward models (GenEval, OCR/PaddleOCR, UnifiedReward, DeQA) require separate conda environments. See [reward-server](https://github.com/yifan123/reward-server).

## Running Training

All training uses `accelerate launch` with config flags:

```bash
# Single GPU
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml \
  --num_processes=1 --main_process_port 29501 \
  scripts/train_sd3.py --config config/grpo.py:general_ocr_sd3_1gpu

# Multi-node (pass rank as argument)
bash scripts/multi_node/sd3.sh 0
```

Key training scripts: `train_sd3.py`, `train_sd3_fast.py`, `train_flux.py`, `train_qwenimage.py`, `train_wan2_1.py`, `train_bagel.py`.

## Project Structure

```
config/          # ml_collections configs (base.py, grpo.py, dpo.py, sft.py, grpo_guard.py)
flow_grpo/       # Core library
  diffusers_patch/  # Modified diffusers pipelines with log-prob computation
  rewards.py        # Reward factory functions (12+ reward models)
  stat_tracking.py  # Per-prompt advantage normalization
  ema.py            # EMA wrapper
  prompts.py        # Prompt generation functions
  *_scorer.py       # Individual reward model wrappers
scripts/         # Training scripts + shell launchers + demos
dataset/         # Prompt datasets (OCR, PickScore, GenEval, etc.)
```

## Code Conventions

- **Config**: `ml_collections.ConfigDict()` with `absl.flags`. Configs are Python functions in `config/` that return ConfigDict objects. Selected via `--config config/grpo.py:function_name`.
- **Naming**: PascalCase for classes, snake_case for functions. Private functions prefixed with `_`.
- **Reward functions**: Factory pattern returning closures: `def reward_fn(device): ... return _fn`
- **Logging**: `wandb` for experiment tracking, `accelerate.logging.get_logger()` for console.
- **Formatting**: `black` (available in dev deps).
- **Type hints**: Minimal; not enforced.

## Important Technical Details

- **Precision**: Prefer fp16 over bf16 for SD3 (smaller log-prob errors). FLUX and Wan require bf16.
- **Batch size matters**: SD3 output differs with different batch sizes; must match between collection and training.
- **On-policy verification**: Set `num_batches_per_epoch=1, gradient_accumulation_steps=1` to verify ratio=1.
- **Hyperparameter formula**: `group_number = train_batch_size * num_gpu / num_image_per_prompt * num_batches_per_epoch`. Empirically: group_number=48, group_size=24.
- **SDE modes**: `sde` (standard velocity-based noise injection) and `cps` (Coefficients-Preserving Sampling).
- **clip_range**: Use small values (default 1e-4), especially for Flow-GRPO-Fast due to larger log-prob errors at low-noise steps.

## GPU Topology & NCCL Issue (This Server)

This server has 4x A40 48GB with a split NVLink topology:
- GPU 0-1: NVLink x4 (fast)
- GPU 2-3: NVLink x4 (fast)
- Cross-group (0/1 ↔ 2/3): PCIe NODE interconnect (slow)

**NCCL P2P hangs on 4-GPU runs.** When NCCL attempts peer-to-peer direct transfers across the two NVLink groups, communication deadlocks. This does not affect 2-GPU runs within the same NVLink group (e.g., GPU 0+1 or GPU 2+3).

**Fix:** Set `NCCL_P2P_DISABLE=1` to force NCCL to use shared memory (SHM) instead of P2P direct transfers. NCCL still handles all distributed communication (all_reduce, barrier, etc.), just via a different transport path.

```bash
# Required for 4-GPU training on this server
NCCL_P2P_DISABLE=1 accelerate launch --num_processes=4 ...
```

## A40 48GB Memory Constraints

The original configs target 80GB GPUs (A100/H100). On A40 48GB:
- **Resolution 512 + batch_size=4 + LoRA + no CFG**: fits (~42GB/card)
- **Resolution 512 + batch_size=8**: OOM on single card
- Disable CFG during training (`config.train.cfg = False`) to save memory (also acts as CFG distillation per the paper)
- Test configs added: `pickscore_sd3_test_1gpu`, `pickscore_sd3_test_2gpu`, `pickscore_sd3_test_4gpu`

## 测试

`tests/test_prior.py` 包含 Prior Shaping 模块的单元测试（34 个）。运行方式：

```bash
python -m pytest tests/test_prior.py -v
```

其余模块无正式单元测试。验证通过 W&B 的 reward 曲线和图像质量指标。`scripts/demo/` 中的脚本可用于 sanity check。

## Prior Shaping（先验分布优化）

与 Flow-GRPO 不同，Prior Shaping **固定 DiT 权重**，通过优化噪声先验分布来提升生成质量。利用 Flow Matching 的 ODE 特性：固定噪声 → 固定图像 → 固定 reward，因此噪声-reward 映射可永久缓存。

### 运行方式

```bash
# 单卡
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml \
  --num_processes=1 --main_process_port 29501 \
  scripts/prior_shaping.py --config config/prior_shaping.py:pickscore_sd3_prior_1gpu

# 4卡（本服务器需要 NCCL_P2P_DISABLE=1）
NCCL_P2P_DISABLE=1 accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml \
  --num_processes=4 --main_process_port 29501 \
  scripts/prior_shaping.py --config config/prior_shaping.py:pickscore_sd3_prior_4gpu

# 4卡快速 smoke test（2 epoch，5 步）
NCCL_P2P_DISABLE=1 accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml \
  --num_processes=4 --main_process_port 29501 \
  scripts/prior_shaping.py --config config/prior_shaping.py:pickscore_sd3_prior_4gpu_smoke
```

### 三种先验优化策略

通过 `config.prior.update_method` 切换：

| 策略 | 值 | 说明 |
|------|------|------|
| **Reward-Weighted** | `"reward_weighted"` | 默认。On-policy，所有样本通过 advantage 加权贡献（类似 GRPO）。用对角高斯拟合。 |
| **CEM** | `"cem"` | Cross-Entropy Method。从历史缓存中选 elite 样本拟合高斯。 |
| **Particle** | `"particle"` | 非参数化。按 reward 加权从噪声 buffer 重采样 + 高斯扰动。可表示任意复杂的多峰分布。 |

### 关键配置参数

```python
# 通用参数
config.prior.update_method = "reward_weighted"  # "reward_weighted" / "cem" / "particle"
config.prior.temperature = 1.0                   # softmax 温度，控制 exploitation vs exploration
config.prior.cache_dir = "cache/prior_shaping"   # 噪声-reward 缓存目录（所有策略都会保存）

# Gaussian 方法专用（reward_weighted / cem）
config.prior.regularization_mode = "kl"          # "kl"（默认）或 "interpolation"
config.prior.kl_max = 1.0                        # KL(p_shaped || N(0,I)) 的上限
config.prior.alpha = 0.1                         # 插值强度（interpolation 模式）

# CEM 专用
config.prior.elite_ratio = 0.1                   # 选取 top 10% 作为 elite
config.prior.use_cache_history = True             # 是否使用历史缓存数据
config.prior.max_cache_epochs = 50                # 最多加载最近 N 个 epoch 的缓存

# Particle 专用
config.prior.perturbation_std = 0.1              # 重采样后的高斯扰动 σ
config.prior.mix_ratio = 0.1                     # 10% 的样本从 N(0,I) 采样用于全局探索
```

### 项目结构（Prior Shaping 相关）

```
config/prior_shaping.py     # 实验配置（PickScore 1/2/4 GPU + smoke test）
flow_grpo/prior.py          # GaussianPrior、ParticlePrior、RewardCache
scripts/prior_shaping.py    # 主脚本（镜像 train_sd3.py 结构）
tests/test_prior.py         # 单元测试（34 个）
cache/                      # 噪声-reward 缓存（.gitignore）
```

## Prior Policy Network（轻量策略网络）

与 Prior Shaping（直接修改分布参数）不同，Prior Policy 训练一个小型神经网络 `π_φ(z|prompt)` 来输出 prompt-conditioned 的噪声分布。DiT 完全冻结，只训练这个轻量网络（~2M 参数）。

### 核心区别

```
Flow-GRPO:     z ~ N(0,I) → DiT_θ(z, prompt) → image    训练 DiT（LoRA）
Prior Policy:  z ~ π_φ(·|prompt) → DiT(z, prompt) → image  训练 policy network
```

### 运行方式

```bash
# 单卡
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml \
  --num_processes=1 --main_process_port 29501 \
  scripts/train_prior_policy.py --config config/prior_policy.py:pickscore_sd3_policy_1gpu

# 4卡
NCCL_P2P_DISABLE=1 accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml \
  --num_processes=4 --main_process_port 29501 \
  scripts/train_prior_policy.py --config config/prior_policy.py:pickscore_sd3_policy_4gpu

# 8卡 H20
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml \
  --num_processes=8 --main_process_port 29501 \
  scripts/train_prior_policy.py --config config/prior_policy.py:pickscore_sd3_policy_8gpu_h20
```

### 训练方式：Advantage-Weighted Regression

```python
loss = -mean(advantage_i * log_prob_per_dim_i)
# advantage 可以是负的：正 advantage → 增加该 noise 的概率，负 advantage → 远离该 noise
# log_prob 用 mean（非 sum）避免高维梯度爆炸
# 不使用 softmax（会把负 advantage 变成正的，丢失"远离坏点"的信号）
```

### 关键配置参数

```python
config.policy.type = "gaussian"          # 策略类型："gaussian" 或 "transformer"
config.policy.hidden_dim = 512           # 网络隐藏层维度
config.policy.learning_rate = 1e-3       # 学习率（log_prob 用 mean 后梯度较小，需要较大 lr）
config.policy.temperature = 1.0          # advantage 缩放温度
config.policy.kl_weight = 0.01           # KL(π||N(0,I)) 正则化，防止方差塌缩
config.policy.train_every_n_epochs = 1   # 每 N epoch 训练一次
```

### 项目结构（Prior Policy 相关）

```
config/prior_policy.py        # 实验配置（1/4/8 GPU + smoke test）
flow_grpo/prior_policy.py     # GaussianPolicy 网络定义 + AWR loss
scripts/train_prior_policy.py # 主训练脚本
tests/test_prior_policy.py    # 单元测试（15 个）
```
