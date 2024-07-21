---
title: "Pytorch DDP"
date: 2023-05-01T15:00:00-08:00
categories:
  - blog
tags:
  - Pytorch
  - Python
  - Distribution
---

## DDP 是个啥

### python GIL

python 是通过引用计数做内存管理。[GIL机制](https://realpython.com/python-gil/﻿) 是所有线程共享一把锁。对于 multi-thread process，锁竞争大，效率不高。

### 古早DP

```python
model = torch.nn.DataParallel(model)
```

单进程多线程（受GIL限制），Parameter Server架构(PS主从模式)：server节点*scatter*参数到worker节点，*gather*接受所有worker节点计算的梯度，求和(all reduce)之后再scatter到worker节点。

![图片](/assets/images/bio-photo.jpg)

DP中的一些通信语义是通过两卡间p2p通信实现的。

```cpp
std::vector<at::Tensor> scatter﻿ (
  const at::Tensor& tensor,
  at::IntArrayRef devices,
  const c10::optional<std::vector<﻿int64_t﻿>>﻿& chunk_sizes,
  int64_t dim,
  const c10::optional<std::vector<c10::optional<at::cuda::CUDAStream>>﻿>﻿& streams) {
  TORCH_CHECK﻿(﻿!devices.﻿empty﻿(﻿)﻿ , "Expected at least one device to scatter to"﻿)﻿;
  if (chunk_sizes.﻿has_value﻿(﻿)﻿) {
    TORCH_CHECK﻿(
      chunk_sizes->﻿size﻿(﻿) == devices.﻿size﻿(﻿)﻿,
      "Expected devices and chunk_sizes to be of same length, but got "
      "len(devices) = "﻿ ,
      devices.﻿size﻿(﻿)﻿ ,
      " and len(chunk_sizes) = "﻿ ,
      chunk_sizes->﻿size﻿(﻿)﻿)﻿;
  }
  dim = at::﻿ maybe_wrap_dim﻿ (dim, tensor)﻿ ;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<at::Tensor> chunks = chunk_sizes
    ? tensor.﻿split_with_sizes﻿(﻿/*split_sizes=*/﻿*chunk_sizes, /*dim=*/dim)
    : tensor.﻿chunk﻿(﻿/*chunks=*/devices.﻿size﻿(﻿)﻿ , /*dim=*/dim)﻿;
  at::cuda::OptionalCUDAStreamGuard cuda_guard;
```
