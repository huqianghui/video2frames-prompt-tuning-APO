# APO 数据集规模选择指南

[English](en-us/dataset-sizing.md) | **中文**

`train` / `val` / `test` 三个 split 需要多大？答案是：大到你关心的分数差异能从
评估噪声中显现出来——再大就是浪费。本文给出针对本项目的具体流程：用已有运行
结果估计噪声、推导所需样本量，并按阶段扩容而不是拍脑袋。

## 1. 为什么大小重要：选择噪声

APO 依据候选 prompt 在 **val split 上的平均 reward** 来选优。任何 `n` 条任务上的
平均值都带有标准误

```
SE = σ / √n
```

其中 `σ` 是单任务 reward 的标准差。比较两个候选 prompt 时，两个平均值之差只有
超过约 `2.8 × SE`（95% 置信度的双样本 z 检验，`√2 × 1.96 ≈ 2.8`）才可信。低于
这个幅度就是掷硬币——APO 会基于噪声"选出"prompt，报告的最佳分数也无法复现。

要检测两个 prompt 之间大小为 `δ` 的真实差异，需要

```
n ≈ 2 × (1.96 × σ / δ)²    （每个 prompt，在 val split 上）
```

以本项目混合 reward 典型的 `σ ≈ 0.20` 计算：

| val 大小 `n` | SE | 能可靠检测的最小差距（≈2.8×SE） |
| --- | --- | --- |
| 24（默认） | ~0.041 | > 0.11 |
| 50 | ~0.028 | > 0.08 |
| 100 | ~0.020 | > 0.055 |
| 200 | ~0.014 | > 0.04 |

对本项目的含义：默认 val=24 时，一个把真实 reward 提升 0.05 的 prompt 修改是
**看不见的**——这与冒烟运行报告的 "seed prompt was never beaten"（0.37 vs 0.37）
一致。prompt 调优的有效收益通常在 0.03–0.10 区间，所以 **64–100** 是现实可用的
val 大小。

## 2. 用已有运行结果估计 σ（零成本）

不要猜 `σ`——单任务 reward 已经在磁盘上：

- `results/eval_<name>.json`（`evaluate.py` 输出）包含每条任务的 reward。
- `results/report.json`（每次 APO 运行后 `generate_report.py` 输出）包含每个
  候选的 `val_rewards`。

对这些 reward 求标准差，就是你的 `σ`。代入上面的公式和你想检测的 `δ`，即得
val 大小。reward 函数或任务模型有任何变更后要重新估计——`σ` 是（模型、reward、
数据）组合的属性，不是常数。

## 3. 不同 split 需要不同的大小

### 为什么不是权重训练的 8:1:1

传统模型训练 train:val:test ≈ 8:1:1 的前提是：train 是梯度下降的"燃料"，
模型消耗的数据量直接决定学习效果，所以 train 占绝对大头；val 只用来挑超参
/早停；test 做一次验收。APO 中三个 split 的消耗方式完全不同：

| Split | 权重训练（8:1:1 的由来） | 本项目 APO |
| --- | --- | --- |
| train | 梯度下降的燃料，每个 epoch 全量消耗，越多学得越好 | 只按 `--gradient-batch-size`（4–8 条）抽小批失败样例给 critic 写文字批评；加量只增加抽样多样性，不增加"学习量" |
| val | 挑超参/早停，用量小 | 每轮为**每个候选**全量打分（如 9 候选 × 100 条）；它的 SE 直接决定 beam 选择的分辨率，是统计上最重的 split |
| test | 最终验收 | 相同——一次性 held-out 验收，规模由最终结论要分辨的效应量决定 |

一句话：8:1:1 优化的是"模型学到多少"，APO 要优化的是"测量有多准"。
train 是采样池、val 是测量仪器、test 是终审。因此 APO 的规模排序通常是
**val ≥ train，test 由效应量单独定**，与 8:1:1 无关、甚至倒挂（本项目实际
使用 80/100/30，val > train）。正确做法不是按比例分，而是按每个 split 的
统计需求逆推：

三个 split 角色不同，扩容方式也不同：

- **`val` —— 最先扩。** 它驱动 APO 内部的候选选择；它的噪声直接导致选错。
  目标是公式算出的 `n`（本项目通常 64–100）。让 `--val-batch-size` 等于 val
  的全量大小，保证每个候选都在同一批任务上打分。
- **`test` —— 第二优先。** 最终的 baseline-vs-tuned 对比是**配对的**：
  `evaluate.py` 用同一批任务跑两个 prompt，所以用单任务 reward *差值*的标准差
  （`σ_d`，通常远小于 `σ`）代入单样本版公式（`n ≈ (1.96 × σ_d / δ)²`）。
  100 条左右通常足以支撑可信的最终结论。
- **`train` —— 一般不用动。** 每次批评只采 `--gradient-batch-size`（默认 4）条
  任务；40 条的池子已有足够多样性。想改善梯度信号，先加
  `--gradient-batch-size` 或 `--beam-rounds`，再考虑加 train 数据。

### 术语："轮"具体指什么（train / val 的实际消耗量）

"每轮抽几条"这类说法容易混淆 run / round / branch 三个层级。精确的层级
关系与抽样时机如下：

```
1 次 run（一次 apo_train.py 执行）
└── beam_rounds 轮
    └── 每轮：beam 中每个存活 prompt（beam_width 个）
        └── 各生成 branch_factor 个子候选
            └── 每个子候选的生成 = 1 次 critique：
                从 train 池抽 gradient_batch_size 条 → rollout
                → critic 看失败样例写批评 → 改写出新 prompt
```

- **train 的抽样发生在"每次 critique"**（即每生成一个子候选），不是每轮
  也不是每次 run。单次 run 的消耗量：

  ```
  critique 次数 ≈ beam_rounds × beam_width × branch_factor
  train rollout 数 = critique 次数 × gradient_batch_size   （可重复抽）
  ```

- **val 的消耗发生在"每个候选"**：每个候选（含种子）都在全量 val 上打分，
  `val rollout 数 = 候选总数 × val_size`。

以 2026-07-17 的 v2 run（beam 2×2×2、gradient batch 8、val 100）为例：
round 1 从种子 v0 生成 v1–v4，round 2 从存活的 v1、v3 各生成 2 个
（v5–v8），共 8 次 critique × 8 条 = **64 条 train rollout**（80 条池子
足够）；而 9 个候选 × 100 条 = **900 条 val rollout**——这就是"val 是
统计上最重的 split"的直接体现，也是 train 池远小于 val 仍然够用的原因。

## 4. 阶梯式扩容：粗筛 + 大集复评

在大 val 上评估每个候选是最贵的环节
（`成本 ≈ beam_rounds × beam_width × branch_factor × (gradient_batch + val_batch)`
条 rollout，每条一次多模态调用 + 一次 judge 调用）。标准解法是
racing / successive-halving 阶梯——多数候选便宜筛，少数候选贵重评：

1. **粗筛** —— 用中小规模的 val（如 24–64）跑 APO。小样本足以淘汰明显差的
   候选；只有接近平手的才会被噪声左右。
2. **复评** —— 训练结束后，从 `results/report.md` 取 top 2–3 个候选 prompt 加上
   baseline，各自在一个更大的 held-out 复评集（100–200 条，从全量 5850 池中
   采样，与 train/val/test 不重叠）上重新评估：

   ```bash
   .venv/bin/python evaluate.py --name baseline
   .venv/bin/python evaluate.py --prompt results/best_prompt.txt --name tuned
   # 以及从 report.md 保存下来的候补 prompt
   ```

   最终赢家以这些分数为准，而不是小 val 的分数。
3. **有证据才升级** —— 如果复评的 best-vs-baseline 差距小于复评集的 `2 × SE`，
   说明效果尚未确立。先加大搜索（更多 beam 轮数/宽度、更好的 gradient batch）；
   只有当一个"有希望但未确认"的差距需要更窄的置信区间时才扩数据集。

这套阶梯不需要改 APO 内部逻辑——`prepare_data.py` 的大小都是 CLI 参数，
`evaluate.py` 接受任意 prompt 文件和 split。

## 5. 采样技术

- **`val`/`test` 保持分层随机**（已是默认行为：`prepare_data.py` 按数据集家族
  分层、固定 seed）。这两个 split 必须代表部署时的分布，绝不能人为加偏。
- **`train` 反而可以偏向难例。** 梯度步从失败中学习，低分任务信息量最大。
  先用 baseline prompt 给候选池打分，再把低分任务多采进 `train`。这通常比
  单纯加（大多简单的）train 数据更有效。
- **采样始终带 `--probe-content-filter`**，让被内容过滤器 block 的视频
  （reward 恒为 0，纯噪声）不进入任何 split；探测结果按视频缓存在
  `data/content_filter_cache.json`，之后扩容只探测新增视频。

## 6. 操作手册：数据与 beam 同步逐级扩容

数据集大小和 beam 超参数是同一份预算——按阶段同步扩，一次动一个轴，用每个
阶段的数字决定下一步。每次运行的 rollout 总成本约为：

```
rollouts ≈ val_size                                   （种子 prompt baseline）
         + beam_rounds × beam_width × branch_factor
           × (gradient_batch_size + val_batch_size)
```

**Stage 0 —— 冒烟（每个环境跑一次）。** 验证的是闭环，不是科学结论。

```bash
.venv/bin/python prepare_data.py --train-size 2 --val-size 2 --test-size 2 --probe-content-filter
.venv/bin/python apo_train.py --smoke
```

进入下一阶段的条件：运行完成、`results/report.md` 正常生成。

**Stage 1 —— 试点：测量噪声。** 默认大小、默认 beam。

```bash
.venv/bin/python prepare_data.py --train-size 40 --val-size 24 --test-size 30 --seed 42 --probe-content-filter
.venv/bin/python apo_train.py                     # beam 2x2x2，gradient batch 4，val batch 24
.venv/bin/python evaluate.py --name baseline      # 单任务 reward -> σ
```

本阶段的产出：从 `results/eval_baseline.json` 得到 `σ`，从 `results/report.md`
看候选 val 分数的分布。决策规则：

- 所有候选分数彼此都在 `2.8 × σ/√24` 之内 → val 太小无法选优；进入 Stage 2。
- 有候选明显胜出 → 可直接在 test 上确认并结束。

**Stage 2 —— 先拓宽搜索，再磨利尺子。** 一次只动一个轴，才能归因收益：

1. *同样数据、更多探索* —— 提高 `--beam-rounds 3`（深度：更多轮批评-改写）或
   `--beam-width 3` / `--branch-factor 3`（广度：更多并行候选）。优先加深度：
   轮数是复利，宽度只是一次性多样性的线性花费。
2. *更好的批评* —— 提高 `--gradient-batch-size 8`，让每次文本梯度看到更多失败
   样例（便宜：只涉及 train rollout）。
3. *更锐利的选择* —— 把 val 扩到第 1 节算出的大小，并同步 flag：

   ```bash
   .venv/bin/python prepare_data.py --train-size 40 --val-size 64 --test-size 100 --seed 42 --probe-content-filter
   .venv/bin/python apo_train.py --beam-rounds 3 --val-batch-size 64
   ```

   探针缓存让扩容很便宜：只有新增的视频会被探测。注意换了大小重新采样会
   重新发牌——train/val/test 仍互不重叠，但单条任务可能换 split，之后要重跑
   baseline 评估。

决策规则：只要 `report.md` 显示后期轮次仍出现 `Best prompt updated`，就继续加大
探索。如果最后一轮从未改进，再加轮数就是浪费钱——停止扩 beam。

**Stage 3 —— 在 held-out 数据上确认。** 用 test / 大复评集给决赛选手复评
（第 4 节第 2 步）：

```bash
.venv/bin/python evaluate.py --name baseline
.venv/bin/python evaluate.py --prompt results/best_prompt.txt --name tuned
```

只有配对差距超过 test 集的 `2 × SE` 才发布调优后的 prompt；否则带着最便宜的
未尝试手段回到 Stage 2。

症状与旋钮对照表：

| 症状（来自 report.md / eval） | 旋钮 | 方向 |
| --- | --- | --- |
| 候选分数在噪声内打平 | `--val-batch-size` + val split 大小 | 扩 val |
| 候选都和种子差不多 | `--branch-factor`、gradient 模型 | 更多/更强的修改 |
| 每轮都刷新 best | `--beam-rounds` | 加轮数 |
| 后期轮次从不改进 | `--beam-rounds` | 停止加 |
| 批评反复说同一个问题 | `--gradient-batch-size`、更难的 train 任务 | 更丰富的失败样例 |
| 最终差距貌似存在但未确认 | test split 大小 | 扩 test |

## 7. 本项目推荐默认值

| Split | 试点（冒烟） | 工作大小 | 何时继续扩 |
| --- | --- | --- | --- |
| train | 4 | 80 | 仅当 gradient batch 看起来重复单调 |
| val | 2 | 100（由实测 σ 定，见第 8 节） | top 候选排名跨运行不稳定 |
| test | 2 | ~100（分辨 0.03–0.04 效应） | 最终配对差距 `p ≈ 0.05`，需要更窄置信区间 |

一句话流程：从 `results/eval_*.json` 估 `σ`，按你关心的差距 `δ` 用
`n ≈ 2(1.96σ/δ)²` 定 val 大小，train 保持小而难，小 val 粗筛，大 held-out 集确认。

## 8. 实测校准与各阶段条数（2026-07-18 更新）

2026-07-17/18 的端到端运行（见
[reward-comparison.md](reward-comparison.md) 第 5–6 节）把第 1 节的
估算换成了实测数字：

- **单任务 reward SD（σ）**：0.113（v1 reward）/ 0.133（v2 reward），来自
  `results/<run_id>/report.json` 各候选的 `val_rewards`——比第 1 节假设的
  0.20 略小。
- **val=100 时候选均值的 SE**：0.011–0.013。两次运行中 9 个候选的均值跨度
  只有 ~0.05，即整个候选群横跨约 4 个 SE，中游候选之间的 beam 选择由噪声
  主导——val=100 在这个任务上是"够用但不富余"的大小。
- **test 配对差值 SD（σ_d）**：0.193（gpt-5.4 探测，30 条配对比较）。n=30
  时配对 SE = 0.035，而本项目的真实效应量都在 0.03–0.04 量级（APO 全程
  +0.035、换 target 模型 +0.041）——**30 条 test 分辨不了它们**，这就是
  reward-comparison 里反复出现"在噪声内 / 尚不足以下结论"的原因。要让
  0.04 的效应达到 2 个 SE，需要 n ≈ (0.193 / 0.02)² ≈ 90–100 条。

据此，各阶段的推荐条数：

| 阶段 | train | val | test | 依据 |
| --- | --- | --- | --- | --- |
| Stage 0 冒烟 | 2 | 2 | 2 | 只验流程，不看分数 |
| Stage 1 试点 | 40 | 24 | 30 | 目的是测 σ，不是出结论 |
| Stage 2 正式训练（当前） | 80 | 100 | 30 | val SE ≈ 0.012 支撑 beam 选择 |
| Stage 3 终态（下次重采样） | 80 | 100 | **~100** | 分辨 0.03–0.04 效应需配对 SE ≤ 0.02 |

终态 80:100:100，作为一般性经验比例可记成 **train : val : test ≈ 2 : 4 : 4**
——与 8:1:1 接近倒挂。注意这只是相对大小的比，不是对总量的划分（候选池
5850 条，只按需抽 280 条）；且预算增加时增量应先给 val/test、train 基本
不动，所以规模越大比例越偏离这个起点。两点操作提醒：

1. **test 扩容要一次到位、之后冻结。** 重新采样会重新发牌，所有已记录的
   baseline 锚点（reward-comparison 第 5/6 节的 0.5686、0.6095 等）随之
   作废、需要重测。不要分多次小步扩 test。
2. **重复评估与扩 test 互补而不互替。** 同一 test split 跑 2–3 遍取平均，
   可把生成/judge 方差砍到 1/√k，成本远低于扩任务数；但它压不住"任务只有
   30 条"的采样噪声。关键结论应同时依赖 ~100 条 test 和 2–3 遍重复评估。
