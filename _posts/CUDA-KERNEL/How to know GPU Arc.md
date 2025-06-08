---
title: "如何查询GPU架构与TFLOPS"
date: 2025-06-08
categories: [CUDA, GPU]
tags: [GPU, CUDA, 性能, TFLOPS]
---

# 如何查询GPU架构与TFLOPS

## 1. 查询GPU型号

在终端输入以下命令，查看当前GPU型号：

```bash
nvidia-smi
```

## 2. 查找官方参数

访问 [NVIDIA 官网](https://www.nvidia.com/) 或 [TechPowerUp GPU Database](https://www.techpowerup.com/gpu-specs/) 搜索你的GPU型号，获取CUDA核心数和主频等参数。

比如 我的GPU是RTX 4070 Super, https://www.techpowerup.com/gpu-specs/nvidia-ad104.g1013 
查询结果如下：

![RTX 4070 Super](sherry-1001.github.io/_images/4070_super.png)


## 3. 计算理论TFLOPS

理论FP32 TFLOPS计算公式：

```
TFLOPS = CUDA核心数 × Boost频率(Hz) × 2 / 1e12
```

示例（RTX 3090）：

- CUDA核心数：10496
- Boost频率：1.7 GHz = 1.7e9 Hz

计算：

```
TFLOPS = 10496 × 1.7e9 × 2 / 1e12 ≈ 35.7 TFLOPS
```

## 4. 直接查表

可直接在[NVIDIA官方文档](https://resources.nvidia.com/en-us-tensor-core/)或TechPowerUp等网站查找主流GPU的理论TFLOPS。

---