# 性能调优：并发配置与运行时长估算

[English](performance-tuning.md) | **中文**

在性能较好的 Linux VM 和高 quota 的 Azure OpenAI 部署上，如何让
`apo_train.py` 跑得更快，以及如何用公式计算参数而不是靠猜。

## 并发模型

APO 调度 rollout 的方式是「**批内并行、批间串行**」：

1. 每次评估（一个 prompt × 一批任务）会把整批 rollout 入队，由
   `--n-runners` 个 runner 进程并行消费。有效并行度 =
   `min(n_runners, 批大小)`。
2. 算法主循环本身是串行的：每个 parent 的梯度评估、每个 candidate 的
   验证评估，都是依次进行。
3. 每生成一个新 candidate 还需要 2 次**串行** LLM 调用（text gradient +
   apply edit），任何 runner 设置都无法并行化它们。
4. 单条 rollout 内部，多模态分析调用和 judge 调用也是串行的——一条
   rollout 会占用一个 runner 直到结束。

## 公式

设：

| 符号 | CLI 参数 | 默认值 |
| --- | --- | --- |
| `R` | `--beam-rounds` | 2 |
| `W` | `--beam-width` | 2 |
| `B` | `--branch-factor` | 2 |
| `g` | `--gradient-batch-size` | 4 |
| `v` | `--val-batch-size` | 24 |
| `n` | `--n-runners` | 4 |
| `t` | — | 实测单条 rollout 耗时（实践中约 6 秒） |

**单次运行的 rollout 总数**（种子 prompt 初始验证 + 每轮的梯度评估与
验证评估）：

```
N = v + R × W × B × (g + v)
```

**墙钟时间估算：**

```
T ≈ t × [ ceil(v/n) × (1 + R·W·B)          # 种子 + 各候选的验证
        + R·W·B × ceil(g/n) ]              # 梯度评估
  + R × W × B × (t_gradient + t_edit)      # 串行 meta-prompt 调用
```

用 `gpt-4.1` / `gpt-4.1-mini` 时，`t_gradient + t_edit` 通常为 20–40 秒。

**示例（默认参数，n = 4）：** N = 24 + 2·2·2·(4+24) = 248 条 rollout；
T ≈ 6·(6·9 + 8·1) + 8·30 ≈ 10–12 分钟。

**示例（n = 12, g = 8）：** 每个候选的验证从 6 波降到 2 波；
T ≈ 6·(2·9 + 8·1) + 8·30 ≈ 6–7 分钟，此时串行的 meta-prompt 调用
成为主要开销。

## 如何选择 `n_runners`

- **受批大小限制：** 超过 `min(g, v)` 的 runner 在对应阶段会闲置。
  `v = 24` 时，`n` 超过 24 没有任何收益。
- **受 Azure OpenAI quota 限制：** 每条进行中的 rollout 占用一个请求
  （帧图片的 token 开销很大）。所需容量大致为：

  ```
  RPM ≈ n × 60 / t × 2        # ×2：分析调用 + judge 调用
  TPM ≈ n × 60 / t × 单条rollout的token数
  ```

  一条含 10–20 帧的 rollout 约消耗 1–2 万输入 token，因此 `n = 12`、
  `t = 6s` 时，多模态部署需要约 150–250 万 TPM。提高 `n` 前请先核对
  部署 quota。
- **CPU 很少是瓶颈：** runner 大部分时间在等 API 响应，`n` 可以超过
  VM 的核数。

**大 VM + 高 quota 的推荐起点：**

```bash
.venv/bin/python apo_train.py --n-runners 12 --gradient-batch-size 8
```

调大 `g` 既能在梯度阶段填满 runner，也能让 critique 模型每次看到更多
样本。

## Runner 增加到何时失效

当 `n ≥ v` 后，剩余耗时由每轮 `R × W × B × 2` 次串行的
gradient/apply-edit 调用主导（默认参数下约 2 分钟/轮）。降低 `R`、`W`、
`B` 可以线性缩短这部分——代价是搜索广度变小（权衡见
[dataset-sizing.zh.md](dataset-sizing.zh.md)）。要并行化各分支的 meta
调用，需要修改上游 APO 实现（`agentlightning/algorithm/apo/apo.py` 的
`_generate_candidate_prompts`，目前对分支逐个 `await`）。

## 速查表

| 目标 | 改法 |
| --- | --- |
| 加快验证阶段 | `--n-runners` 提到 `min(v, quota 上限)` |
| 梯度阶段不闲置 runner | `--gradient-batch-size ≈ n_runners` |
| 整体更快更省 | 降低 `R`、`W`、`B`（减少候选数） |
| 统计更稳但更慢 | 增大 `v`（见 dataset-sizing.zh.md） |
