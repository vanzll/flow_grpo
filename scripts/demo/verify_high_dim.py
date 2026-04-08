"""验证高维空间的两个反直觉现象：

1. N(0, I) 的点集中在薄球壳上（范数几乎恒定）
2. 任意两个随机向量几乎正交（夹角接近 90°）

Usage:
    python scripts/demo/verify_high_dim.py
"""

import numpy as np

dims = [2, 10, 100, 1000, 10000, 65536]
num_samples = 1000

print("=" * 70)
print("现象 1：N(0, I) 的范数集中在薄球壳上")
print("=" * 70)
print()
print(f"{'维度':>8s} | {'理论均值 √d':>12s} | {'实测均值':>10s} | {'实测 std':>10s} | {'变异系数':>10s}")
print("-" * 60)

for d in dims:
    samples = np.random.randn(num_samples, d)
    norms = np.linalg.norm(samples, axis=1)
    theory_mean = np.sqrt(d)
    cv = norms.std() / norms.mean() * 100
    print(f"{d:>8d} | {theory_mean:>12.2f} | {norms.mean():>10.2f} | {norms.std():>10.4f} | {cv:>9.2f}%")

print()
print("结论：维度越高，变异系数越小，所有点的范数越接近 √d")

print()
print("=" * 70)
print("现象 2：两个随机向量几乎正交")
print("=" * 70)
print()
print(f"{'维度':>8s} | {'理论 cos θ':>12s} | {'实测 cos θ':>12s} | {'实测角度':>10s} | {'cos θ std':>10s}")
print("-" * 65)

for d in dims:
    cos_thetas = []
    for _ in range(num_samples):
        x = np.random.randn(d)
        y = np.random.randn(d)
        cos_theta = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        cos_thetas.append(cos_theta)
    cos_thetas = np.array(cos_thetas)
    angles = np.degrees(np.arccos(np.clip(cos_thetas, -1, 1)))
    theory_cos = 1 / np.sqrt(d)
    print(f"{d:>8d} | {theory_cos:>12.6f} | {cos_thetas.mean():>12.6f} | {angles.mean():>9.2f}° | {cos_thetas.std():>10.6f}")

print()
print("结论：维度越高，cos θ 越接近 0（角度越接近 90°）")
print("      d=65536 时，任意两个向量的夹角 ≈ 89.8°，几乎完全正交")
