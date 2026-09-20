# Shared 与 Last-exited KV：语义与计算示例

本文解释 vllm-lt 中 `shared` 和 `last_exited` 两种 KV 布局，并用可手算的例子展示 Attention 如何产生不同结果。说明基于 2026-09-20 查看时的实现。

## 1. 核心区别

当前 token 在第几轮循环，应该读取历史 token 的哪一轮 KV？

- **last_exited**：按 loop 保存 KV。读取历史 token 的同轮 KV；如果历史 token 已提前退出，则读取由其最后执行轮次复制补齐的 KV。
- **shared**：每个 token、每个物理 layer 只保存一份 KV，不同 loop 覆盖同一位置。后续 token 在任意 loop 都读取历史 token 最终留下的 KV。

这里的 shared 是同一请求、同一 token position、同一物理 layer 在 loop 之间共享存储；不是不同请求、不同 token 或不同 layer 混用内容。

三个维度需要区分：

| 维度 | 含义 |
| --- | --- |
| position | 序列中的 token 位置 |
| loop / depth | token 第几次经过共享 Transformer |
| layer | 一轮 Transformer 内的物理层 |

本文 loop 从 1 开始编号；代码中的 depth 从 0 开始。

## 2. 两个 token、两轮循环的手算例子

假设模型只有一层 Attention，循环两次，序列为 A、B。先处理 A，再计算 B。

为方便手算：

- 单个 head，Q/K/V 都是标量，缩放因子为 1。
- 省略 RoPE、残差、MLP、输出投影和归一化。
- 直接给定 A 各轮及 B 第一轮的 Q/K/V。
- B 第二轮采用下文定义的简化更新规则。
- 两个 token 均执行两轮，没有 early exit。

**这是用于隔离 KV 读取差异的局部算例，不是完整、参数自洽的 Ouro 模型，也不是模型精度测试。** 给定的 A 值与 B 第一轮值不要求由第二轮使用的简化投影规则生成。

Attention 公式：

$$
\operatorname{Attention}(q,K,V)
=\operatorname{softmax}(qK^\top)V
$$

### 2.1 历史 token A 留下的 KV

假设 A 的两轮计算产生：

| A 的轮次 | K | V |
| --- | ---: | ---: |
| 第 1 轮 | 1 | 10 |
| 第 2 轮 | 2 | 20 |

A 执行完后：

```text
last_exited：
loop 1 保存 (K=1, V=10)
loop 2 保存 (K=2, V=20)

shared：
唯一位置先写入 (1,10)，再被覆盖为 (2,20)
最终只保留 (K=2, V=20)
```

### 2.2 B 第一轮：last_exited

给定 B 第一轮：

```text
Q_B¹ = 1
K_B¹ = 1
V_B¹ = 30
```

先写 B 自己的 KV，再执行包含自身位置的因果 Attention。本轮读取：

| 位置 | K | V |
| --- | ---: | ---: |
| A，第 1 轮 | 1 | 10 |
| B，第 1 轮 | 1 | 30 |

分数：

$$
s_A=1\times1=1,\qquad s_B=1\times1=1
$$

权重：

$$
(w_A,w_B)=\operatorname{softmax}([1,1])=(0.5,0.5)
$$

输出：

$$
o_B^1=0.5\times10+0.5\times30=20
$$

### 2.3 B 第一轮：shared

B 写入自己的位置，不覆盖 A。本轮缓存为：

| 位置 | K | V |
| --- | ---: | ---: |
| A，最后留下的第 2 轮 | 2 | 20 |
| B，当前第 1 轮 | 1 | 30 |

分数：

$$
s_A=1\times2=2,\qquad s_B=1\times1=1
$$

权重：

$$
w_A=\frac{e^2}{e^2+e^1}\approx0.7311,\qquad
w_B=\frac{e^1}{e^2+e^1}\approx0.2689
$$

输出：

$$
o_B^1=0.7311\times20+0.2689\times30\approx22.6894
$$

第一轮已经产生差异：

| 布局 | B 读取的历史 | Attention 输出 |
| --- | --- | ---: |
| last_exited | A 第一轮 KV | 20 |
| shared | A 最终第二轮 KV | 22.6894 |

### 2.4 B 第二轮：差异如何继续传播

为演示后续传播，定义 B 第二轮的简化更新：

```text
h = B 第一轮 Attention 输出
Q = h / 20
K = h / 20
V = h
```

两种路径使用同一规则。这不是 Ouro 的真实更新公式。

**last_exited 路径**

上一轮输出为 20：

```text
Q_B² = 1
K_B² = 1
V_B² = 20
```

第二轮读取 A 第二轮 KV，并写入 B 第二轮 KV：

```text
A：(2,20)
B：(1,20)
```

分数为 [2,1]，但两个 V 都为 20，所以：

$$
o_B^2=20
$$

最终存储：

```text
loop 1：A=(1,10)，B=(1,30)
loop 2：A=(2,20)，B=(1,20)
```

**shared 路径**

上一轮输出约为 22.6894：

```text
Q_B² ≈ 1.1345
K_B² ≈ 1.1345
V_B² ≈ 22.6894
```

B 的第一轮 KV 被覆盖，缓存变为：

```text
A：(2,20)
B：(1.1345,22.6894)
```

使用未舍入的中间值计算：

$$
s_A\approx2.2689,\qquad s_B\approx1.2870
$$

$$
(w_A,w_B)\approx(0.7275,0.2725)
$$

$$
o_B^2\approx0.7275\times20+0.2725\times22.6894\approx20.73
$$

最终只保留：

```text
A：(2,20)
B：(1.1345,22.6894)
```

实际模型中，Attention 输出会经过输出投影、残差、MLP、归一化等计算，再影响后续 layer 和 loop。因此第一轮历史读取的差异可以继续改变后续 Q/K/V、gate、退出深度和最终 logits。

## 3. 历史 token 提前退出时怎么读

假设最多 4 轮，A 在第 2 轮退出：

```text
A 第 1 轮：(1,10)
A 第 2 轮：(2,20)
```

B 各轮读取 A 的内容：

| B 的轮次 | last_exited | shared |
| --- | --- | --- |
| 第 1 轮 | (1,10) | (2,20) |
| 第 2 轮 | (2,20) | (2,20) |
| 第 3 轮 | (2,20)，退出时复制补齐 | (2,20) |
| 第 4 轮 | (2,20)，退出时复制补齐 | (2,20) |

对于已经完成的历史 token，设其退出轮次为 e、当前查询轮次为 d：

- last_exited 读取历史 token 的第 min(d,e) 轮 KV。
- shared 读取历史 token 的第 e 轮 KV。

这里的公式描述各自路径的版本选择，不表示两种路径对应轮次的 KV 数值必然相同。后面几轮即使读到相同的给定 A 值，B 的 hidden 也可能已经因第一轮的差异而不同。

## 4. 对应代码实现

### 4.1 存储与页表

见 [KVCacheManager](../vllm_lt/core/kv_cache_manager.py) 的初始化、`allocate()`、`_plane()` 和 `get_block_table()`。

```python
storage_depths = max_loops if layout == "last_exited" else 1

def _plane(depth):
    return depth if layout == "last_exited" else 0
```

K/V 分别存储在以下形状的 tensor 中：

```text
[num_blocks, num_layers, block_size, num_kv_heads, head_dim]
```

物理 tensor 没有显式 loop 维度。last_exited 通过不同 loop 的页表映射到不同物理块，shared 将所有 loop 映射到同一个存储平面。

地址选择可简化为：

```python
plane = depth if layout == "last_exited" else 0
logical_page = position // block_size
offset = position % block_size
physical_block = allocation.block_tables[plane][logical_page]
key_cache[physical_block, layer, offset] = current_k
value_cache[physical_block, layer, offset] = current_v
```

### 4.2 写入与 Attention

`_write_prepared()` 写当前 token 的 KV，`_attend_prepared()` 使用对应页表读取到当前 position 为止的上下文。

last_exited 每次 Attention 只读本轮对应的 KV 平面，**不会把全部 loop 拼成更长的上下文**。上下文长度仍按 token position 计算。

### 4.3 Early-exit 补齐

`finalize_token()` 检查最后执行轮次的各层 KV 已写入后：

- shared 直接返回，唯一位置已保留最后一轮的值。
- last_exited 将该 token 在每个物理 layer 的 K/V 复制到所有未执行深度，并登记有效位置。

复制的是各 layer 的 KV，不是最终 hidden state，也不是整个物理页。异步执行的 [ModelRunner](../vllm_lt/worker/model_runner.py) 还可通过 `finalize_many()` 和 `finalize_kernel` 批量执行补齐，并通过 event 保证时序。

## 5. Prefill 为什么采用不同顺序

当前 [ModelRunner._prefill()](../vllm_lt/worker/model_runner.py) 为两种布局使用不同执行顺序。

last_exited 可以在一个 chunk 内打包多个 prompt token：

```text
A、B、C 一起跑 loop 1
A、B、C 一起跑 loop 2
A、B、C 一起跑 loop 3
A、B、C 一起跑 loop 4
```

每轮读取对应深度的 KV，并保持因果掩码。

shared 为保证每个 token 读取前面 token 的最终 KV，以及结果不依赖 chunk 划分，同一请求按位置推进：

```text
A：loop 1 → 2 → 3 → 4
B：loop 1 → 2 → 3 → 4
C：loop 1 → 2 → 3 → 4
```

不同请求仍可一起 batch。如果将 shared 改成同一 prompt 内所有 token 按轮打包，B 第一轮就可能读到 A 第一轮的 KV，而不是 A 最终 KV；这会改变当前语义。

## 6. 容量与功能差异

请求需要覆盖 T 个 token 位置、页大小为 B、最大循环数为 D 时：

```text
last_exited：ceil(T/B) × D 个物理块
shared：     ceil(T/B) 个物理块
```

| 项目 | last_exited | shared |
| --- | --- | --- |
| 历史版本 | 每轮独立保留；未执行轮次复制退出轮次 | 只保留最终版本 |
| Early-exit 补齐 | 需要 | 不需要 |
| 同请求 prefill token 打包 | 当前支持 | 当前逐位置执行 |
| Prefix caching | 当前支持 | 当前配置校验拒绝 |
| 最大 D 轮的请求块需求 | shared 的 D 倍 | 一份存储平面 |

当前缓存池按 `num_blocks` 预分配。相同 `num_blocks` 下，两种布局分配的 KV tensor 字节数相同；shared 的优势表现为同一个池可容纳更多 token。last_exited 提前退出并不会自动释放对应的深层存储，因为这些位置仍需保存补齐后的历史。

shared 减少存储需求和补齐复制，但当前 prefill 的同请求位置串行化可能影响性能，不能仅凭布局断言一定更快。两种布局改变 Attention 的历史输入，不能作为无损互换的内存优化。

