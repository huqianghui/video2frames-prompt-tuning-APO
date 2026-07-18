# 性能调优：Beam 超参数、并发与运行时长估算

[English](en-us/performance-tuning.md) | **中文**

成本、执行时间、搜索效果是同一枚硬币的不同面：`--beam-rounds` /
`--beam-width` / `--branch-factor` 决定搜索的形状（也就决定了 rollout
总量 = API 成本），`--n-runners` 决定这些 rollout 被并行摊薄的程度，
Azure OpenAI quota 决定并行度的上限。本文把这三层放在一起讲：每个
参数的意义、相互关系、成本公式、`n_runners` 与 quota 的计算方法，
以及运行时长估算。（验证集规模与超参的同步扩容见
[dataset-sizing.md](dataset-sizing.md)。）

## 心智模型：prompt 版本树上的束搜索

把 prompt 优化看成在一棵"prompt 版本树"上搜索：种子 prompt v0 是根，
每个新候选是某个父本的改写。三个 beam 超参数分别控制这棵树的
**深度、存活宽度、分叉数**。

### `--beam-rounds`（R，深度）——迭代多少轮

每一轮 = "对现有好 prompt 做一批改写 → 验证集打分 → 淘汰"。轮数越多，
改进可以**跨轮叠加**——例如 best prompt 的派生链 v0 → v4 → v7 就是
两轮各前进一步的结果。R 决定优化能走多"远"。

### `--beam-width`（W，宽度）——每轮留下几个存活者

每轮打分后，只有验证分最高的 W 个 prompt 存活到下一轮当"父本"。
W = 1 退化为贪心爬山（只沿最好的一条路走，容易困在局部最优）；
W 越大，同时保留的"备选进化路线"越多。

### `--branch-factor`（B，分叉）——每个存活者生成几个后代

对每个父本：跑 `g`（`--gradient-batch-size`）条 train rollout →
gradient 模型写批评 → apply-edit 模型据此改写出 **B 个不同的**新
prompt。B 越大，单个方向上尝试的改法越多样。

### 一轮的完整流程与相互关系

每轮固定三步，各步用的数据集不同（源码见
`agentlightning/algorithm/apo/apo.py`）：

```
第 r 轮：
  ① 生成候选（beam 中 W 个父本，每个 × B 次）
     └─ 从 train 抽 g 条（--gradient-batch-size，apo.py:650）
        → 跑 rollout 拿 trace → gradient 模型写批评 → 改写出新 prompt
        【train 的作用：给 critic 提供"错题样本"，分数不用于任何排名】
        （共 W×B 个新候选，每个额外花 2 次串行 meta 调用）

  ② beam 打分 / 选择
     └─ 从 val 抽一批（--val-batch-size，apo.py:719）
        → 所有候选打分 → 排序 → top-W 存活进入第 r+1 轮
        【val 的作用：决定谁活下来】

  ③ best 复评
     └─ 完整 val 数据集（apo.py:777）
        → 本轮 beam 第 1 名重考 → 复评分赢过历史 best 才更新 best
        【作用：噪声闸门——候选必须两次拿高分才能当 best】
```

关于 ③ 的三个容易踩的点：

- **只有每轮的 beam 第 1 名有资格挑战 best**。如果真正最好的 prompt
  在②被噪声压到第 2 名，它连复评的机会都没有。
- **最终 best 分数来自复评，不是②的初评**——所以 tree.md 里会出现
  "某候选初评 0.756 却不是 best"的情况：它复评时分数掉下来了，说明
  初评里有噪声虚高。
- **`--val-batch-size` < val 全量时，②和③用的数据不同**：②用随机
  子集（洗牌 + 按 epoch 轮转，每条任务一个 epoch 内最多出现一次），
  ③用全量。两者相等时（如 24/24 或 100/100），③退化为同一批题重考
  一遍，只防 LLM 随机性、不提供新题检验。

- **W × B 是每轮的"探索预算"。** W=1, B=4 和 W=2, B=2 每轮都产出
  4 个候选，但前者把预算押在单一路线的多种改法上，后者保留两条
  独立路线。
- **R 与 W×B 是深度 vs 广度的权衡。** 改进要叠加靠 R；避免押错路线
  靠 W；单轮多样性靠 B。
- W 不要超过每轮的候选供给（上一轮存活者 + 新候选），否则等于
  不做淘汰。

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

## 成本与时长公式

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

rollout 总数 ≈ API 成本，**对 R、W、B 都是线性的**——三个参数一起
翻倍，成本是 4 倍：

| 量 | 公式 | 默认 (2/2/2) | 加深 (4/2/2) | 激进 (4/2/3) |
| --- | --- | --- | --- | --- |
| 新 prompt 数 | R×W×B | 8 | 16 | 24 |
| 串行 meta 调用数 | R×W×B×2 | 16 | 32 | 48 |
| 总 rollout 数 | v + R×W×B×(g+v) | 248 | 536 | 792 |

（v = 24，g = 8 时。）

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

时间上有个不对称：rollout 可以被 `--n-runners` 并行摊薄，但每轮
`W×B×2` 次 gradient/apply-edit 调用是**串行**的——所以在墙钟时间上，
**加 R 比加 B 更贵**。

## 如何选择 `n_runners`

- **受批大小限制：** 超过 `min(g, v)` 的 runner 在对应阶段会闲置。
  `v = 24` 时，`n = 4` 每个候选要跑 `ceil(24/4) = 6` 波验证，`n = 12`
  只要 2 波；`n` 超过 24 没有任何收益。梯度阶段同理——把
  `--gradient-batch-size` 提到 ≈ n_runners，否则该阶段有 runner 闲置。
- **受 Azure OpenAI quota 限制：** 每条进行中的 rollout 对多模态部署
  发 1 个分析请求、对 judge 部署发 1 个 judge 请求。所需容量大致为：

  ```
  多模态部署 TPM ≈ n × (60 / t) × 单条输入 token 数
  多模态部署 RPM ≈ n × (60 / t)
  judge 部署 TPM  ≈ n × (60 / t) × judge 输入 token 数（纯文本，几千/条）
  ```

  一条含 10–20 帧的 rollout 约消耗 1–2 万输入 token（帧图片是大头），
  即多模态部署每个 runner 约需 **15 万 TPM**。提高 `n` 前请先核对
  部署 quota。
- **CPU 很少是瓶颈：** runner 大部分时间在等 API 响应，`n` 可以超过
  VM 的核数。

**实例（多模态部署 2.5 M TPM，judge 部署 3 M TPM）：**

| n-runners | 多模态 TPM 需求 | vs 2.5 M quota |
| --- | --- | --- |
| 4 | ~60 万 | 只用 1/4，浪费 |
| **12** | ~180 万 | **舒适区，推荐** |
| 16 | ~240 万 | 贴上限，帧多的视频可能触发 429 |
| ≥ 24 | — | 超过 v，无收益 |

judge 是纯文本调用，`n = 16` 也只占几十万 TPM，通常不是瓶颈。
gradient/apply-edit 每轮只有 `W×B` 次串行调用，可忽略。偶发 429 时
SDK 会自动退避重试，只是变慢不会失败——想榨干 quota 可以先跑一次，
确认日志无 `429` 再加 `n`。

**大 VM + 高 quota 的推荐起点：**

```bash
.venv/bin/python apo_train.py \
  --beam-rounds 4 --beam-width 2 --branch-factor 2 \
  --n-runners 12 --gradient-batch-size 8
```

调大 `g` 既能在梯度阶段填满 runner，也能让 critique 模型每次看到更多
样本。

## Runner 增加到何时失效

当 `n ≥ v` 后，剩余耗时由每轮 `R × W × B × 2` 次串行的
gradient/apply-edit 调用主导（默认参数下约 2 分钟/轮）。降低 `R`、`W`、
`B` 可以线性缩短这部分——代价是搜索广度变小（权衡见
[dataset-sizing.md](dataset-sizing.md)）。要并行化各分支的 meta
调用，需要修改上游 APO 实现（`agentlightning/algorithm/apo/apo.py` 的
`_generate_candidate_prompts`，目前对分支逐个 `await`）。

## 调参决策表

| 观察到的现象 | 动作 |
| --- | --- |
| best prompt 出现在最后一轮（还没收敛） | 加 R |
| 某轮所有新候选都不如父本（改法多样性不足） | 加 B |
| 多条路线分数接近，怕押错 | 加 W |
| 验证阶段慢 | `--n-runners` 提到 `min(v, quota 上限)` |
| 梯度阶段 runner 闲置 | `--gradient-batch-size ≈ n_runners` |
| 整体更快更省 | 降低 `R`、`W`、`B`（减少候选数） |
| 分数差异小于评估噪声 | 先增大 `v`（见 dataset-sizing.md），别急着调 R/W/B |
