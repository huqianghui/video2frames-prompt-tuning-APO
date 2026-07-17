# Reward v1 vs v2 对比记录

[English](reward-comparison.md) | **中文**

本文记录两个 reward 版本的实测对比。对比分两个层次：

1. **刻度校准（已完成，本文第 2 节）**——同一个 baseline prompt 分别用 v1、v2
   打分。回答"两个 reward 的量纲与排序差多少"。v1 分和 v2 分**不可直接比**
   （公式不同、严格程度不同），任何跨版本的比较都必须先有这一步的基准。
2. **端到端对比（待跑，本文第 4 节）**——分别用 v1、v2 各跑一次完整 APO 训练，
   得到 `best_prompt_v1` / `best_prompt_v2`，再做 **2×2 评估**（两个 prompt ×
   两个 reward）。回答"哪个 reward 能驱动 APO 产出更好的 prompt"。

## 1. 运行配置（刻度校准）

| 项 | 值 |
| --- | --- |
| 日期 | 2026-07-17 |
| Prompt | `data/baseline_prompt.txt`（未调优的种子 prompt） |
| Split | `data/test.jsonl`，30 条，sha256 前缀 `d4993bcbaf1a` |
| 生成模型 | `gpt-4.1-mini`（`AZURE_OPENAI_DEPLOYMENT`） |
| Judge 模型 | `gpt-4.1-mini`（两个版本相同） |
| Reward 配置 | `reward/v1/config.yaml`、`reward/v2/config.yaml` 均为默认值 |
| 执行方式 | macOS 串行（每轮约 4 分钟） |

```bash
.venv/bin/python evaluate.py --name baseline_v1 --reward-version v1
.venv/bin/python evaluate.py --name baseline_v2 --reward-version v2
.venv/bin/python compare_rewards.py results/eval_baseline_v1.json results/eval_baseline_v2.json
```

明细文件（含 join 结果）在本机 `results/` 下（git 不入库）。

## 2. 结果

### 总体与按 family

| | v1 | v2 |
| --- | --- | --- |
| **整体均值（30 条）** | **0.734** | **0.522** |
| Charades（22） | 0.757 | 0.560 |
| NWPU（2） | 0.691 | 0.518 |
| VIRAT（2） | 0.838 | 0.674 |
| project（4） | 0.577 | **0.239** |

### 逐任务（按 B−A 降幅排序）

| task | family | v1 | v2 | Δ(v2−v1) |
| --- | --- | --- | --- | --- |
| 5266 | project | 0.808 | 0.311 | −0.497 |
| 375 | Charades | 0.808 | 0.334 | −0.474 |
| 2577 | Charades | 0.512 | 0.181 | −0.331 |
| 5540 | project | 0.452 | 0.138 | −0.314 |
| 3698 | Charades | 0.644 | 0.337 | −0.307 |
| 3108 | Charades | 0.934 | 0.648 | −0.286 |
| 5142 | project | 0.536 | 0.252 | −0.284 |
| 1799 | Charades | 0.922 | 0.652 | −0.270 |
| 5085 | project | 0.512 | 0.255 | −0.257 |
| 2656 | Charades | 0.808 | 0.608 | −0.200 |
| 3059 | Charades | 0.568 | 0.370 | −0.198 |
| 4902 | NWPU | 0.634 | 0.436 | −0.198 |
| 2444 | Charades | 0.652 | 0.454 | −0.198 |
| 2370 | Charades | 0.868 | 0.678 | −0.190 |
| 482 | Charades | 0.904 | 0.720 | −0.184 |
| 4002 | Charades | 0.628 | 0.444 | −0.184 |
| 569 | Charades | 0.676 | 0.502 | −0.174 |
| 584 | Charades | 0.634 | 0.462 | −0.172 |
| 262 | Charades | 0.604 | 0.432 | −0.172 |
| 4417 | Charades | 0.856 | 0.684 | −0.172 |
| 5730 | VIRAT | 0.868 | 0.700 | −0.168 |
| 898 | Charades | 0.772 | 0.609 | −0.163 |
| 1323 | Charades | 0.544 | 0.381 | −0.163 |
| 5717 | VIRAT | 0.808 | 0.648 | −0.160 |
| 4860 | NWPU | 0.748 | 0.599 | −0.149 |
| 2277 | Charades | 0.808 | 0.663 | −0.145 |
| 991 | Charades | 0.832 | 0.697 | −0.135 |
| 4089 | Charades | 0.856 | 0.738 | −0.118 |
| 2152 | Charades | 0.940 | 0.867 | −0.073 |
| 1735 | Charades | 0.892 | 0.855 | −0.037 |

## 3. 解读

1. **v2 全面更低是设计使然，不代表输出变差。** 分字段 judge 更严、规则合规
   占 0.25 权重、场景/快递错误以乘法砍分（×0.5 / ×0.3 / ×0.2），任何一处
   短板都会直接压低总分。
2. **v2 的区分度显著更大。** project family 被拉到 0.239（v1 下还有
   0.577），最好与最差 family 的差距从 v1 的 0.26 拉大到 v2 的 0.44。对
   APO 而言这意味着更陡的梯度信号——"哪里错了"在分数上更可见。
3. **两个 reward 方向一致。** 30 条任务全部 v2 < v1、排序大体相同，说明 v2
   没有扭曲优化方向，只是更严、更细。
4. **大降幅任务的归因（来自 v2 组件日志）：**
   - `5540` / `2577`：courier 误报硬门触发（×0.3），叠加较低的
     `judge_detail`（0.34 / 0.28）。
   - `5266`：场景硬门触发（×0.5）。注意 v1 那轮运行里 scene 是判对的——
     见下面的噪声说明。
   - `375`：没有硬门；`rule_no_meta_words` 不合规（提到了
     camera/frame/timestamp 类词）+ 分字段 judge 很低（0.18/0.18/0.03）。

**噪声说明（重要）：** 两轮评估各自重新调用生成模型，所以 v1、v2 打分的是
**不同的生成结果**。逐任务的 Δ 混合了"reward 定义差异"和"生成方差"两种
来源（如 `5266` 的场景在 v1 那轮生成里是对的、v2 那轮错了）。**均值层面**
的结论（v2 更严、区分度更大、方向一致）不受影响，但不要过度解读单条任务的
Δ。若需要严格隔离 reward 定义差异，应缓存同一批生成结果、离线重打分。

## 4. 下一步：端到端对比（Linux 高并发）

在 Linux VM（Python ≤ 3.13，`fork` 启动方式）上分别用两个 reward 各跑一次
APO，然后做 2×2 评估：

```bash
# 两次训练（beam/批量参数按 doc/performance-tuning.zh.md 选择；大 VM 示例）
.venv/bin/python apo_train.py --reward-version v1 --n-runners 12 --gradient-batch-size 8
.venv/bin/python apo_train.py --reward-version v2 --n-runners 12 --gradient-batch-size 8
# 记下各自的 results/<run_id>/（results/latest 只指向最近一次！）

# 2×2 评估：两个 best prompt 各在两个 reward 下打分
.venv/bin/python evaluate.py --prompt results/<run_v1>/best_prompt.txt --name tuned_v1_under_v1 --reward-version v1
.venv/bin/python evaluate.py --prompt results/<run_v1>/best_prompt.txt --name tuned_v1_under_v2 --reward-version v2
.venv/bin/python evaluate.py --prompt results/<run_v2>/best_prompt.txt --name tuned_v2_under_v1 --reward-version v1
.venv/bin/python evaluate.py --prompt results/<run_v2>/best_prompt.txt --name tuned_v2_under_v2 --reward-version v2

# 对比（同一 reward 刻度下比较两个 prompt 才有意义）
.venv/bin/python compare_rewards.py results/eval_tuned_v1_under_v1.json results/eval_tuned_v2_under_v1.json
.venv/bin/python compare_rewards.py results/eval_tuned_v1_under_v2.json results/eval_tuned_v2_under_v2.json
```

判读方法：

- 以本文第 2 节的 baseline 数字为基线，看每个 tuned prompt 相对 baseline
  的提升（同一 reward 刻度内比较）。
- `tuned_v2` 若在 **v1 刻度**下也不差于 `tuned_v1`，且在 v2 刻度下明显更好，
  说明 v2 的额外信号（分字段、规则、硬门）产生了真实收益而非过拟合自身公式。
- 分类字段（scene/courier）的准确率可从 `results/eval_*.json` 的组件明细
  或运行日志统计，作为不依赖任一 reward 刻度的客观参照。

结果出来后回填本文第 5 节。

## 5. 端到端对比结果（2026-07-17，Linux VM）

### 运行配置

| 项 | 值 |
| --- | --- |
| 日期 | 2026-07-17（Linux VM，并行 runner） |
| 数据 | `prepare_data.py --train-size 80 --val-size 100 --freeze-test --probe-content-filter`（train 80 / val 100 / test 30） |
| APO 运行（v1 reward） | `results/20260717_143056/` |
| APO 运行（v2 reward） | `results/20260717_145308/` |
| Beam / 批量参数 | 见 VM 上各运行目录的 `summary.json`（未拷贝到本机） |

> **Test split 注意：** VM 上冻结的 `test.jsonl`（sha256 前缀
> `b5065f2d3016`）**不是**第 2 节刻度校准用的那个 split（任务 ID 不同——
> 2282/4221/… vs 5266/375/…，且包含旧 split 没有的 `ucf_crime` 任务）。
> 第 2 节的 baseline 锚点（v1 0.734 / v2 0.522）不适用；因此已在该 split
> 上**重新评估 baseline prompt**（2026-07-18），即下表的锚点行。

### 2×3 矩阵（test 上的 mean_reward，30 条）

| | v1 刻度打分 | v2 刻度打分 |
| --- | --- | --- |
| `baseline`（种子 prompt） | 0.7552 | 0.5686 |
| `tuned_v1`（v1-reward 运行的 best） | 0.7509（−0.004） | 0.5993（+0.031） |
| `tuned_v2`（v2-reward 运行的 best） | **0.7609**（+0.006） | **0.6033**（+0.035） |

（括号内为同刻度下相对 baseline 的差值。）

训练侧数字（val，100 条）：v1 运行种子 0.764 → best 0.798；v2 运行种子
0.594 → best 0.643。两次运行 beam 均为 2/2/2；且**两次运行中所有第 2 轮
子候选的得分都低于其第 1 轮父候选**——即全部收益来自第一次编辑，搜索在
一轮后就饱和了。

`tuned_v2` vs `tuned_v1` 的逐任务胜负：

- v1 刻度：13 胜 / 11 负 / 6 平；v2 刻度：14 胜 / 15 负 / 1 平。
- 按 family 的均值没有一致方向（v1 刻度下 tuned_v2 在 Charades、ucf_crime
  上更好，在 NWPU/VIRAT/project 上更差；v2 刻度下互有胜负）。

### 结论

有了 baseline 锚点之后，结论比"统计平局"更清晰：

- **v1 刻度下什么都没提升**——tuned_v1 −0.004、tuned_v2 +0.006，均在噪声
  内。v1 的粗粒度语义 judge（0.6 权重）看不出任何一个 tuned prompt 的差别。
- **v2 刻度下两个 tuned prompt 都提升约 +0.03**——包括训练时从没见过 v2
  reward 的 `tuned_v1`。真正被改善的是 v2 显式度量、而 v1 几乎不度量的
  部分：规则合规与字段结构（确定性的、prompt 可修复的），而不是描述的
  语义内容。
- 两个 tuned prompt 正面对比仍在噪声内（+0.010 / +0.004，逐任务接近
  抛硬币）。

### 诊断：为什么 v2 把刻度"打开"了，提升还是只有 ~0.04

以下证据来自 `report.json`（每个候选的 val 逐任务 reward 向量）和
gradient 批评文本：

1. **v2 打开的是任务轴，不是 prompt 轴。** v2 baseline 更低（0.57 vs
   0.76）来自对*难任务*的更严打分（分字段 judge、视觉歧义视频上的硬门）。
   这部分余量要靠更强的视觉模型才能吃到，靠改指令措辞吃不到。prompt 编辑
   真正能动的——确定性的规则合规部分（0.25 权重）加边际的 judge 提升——
   大约值 +0.03–0.05，而 APO 恰好拿到了这么多（test 上 +0.035）。
2. **候选间差距 ≈ 测量噪声。** 每次运行 9 个候选的 val 均值跨度约 0.05，
   而单个候选均值的 SE（n=100，逐任务 SD 约 0.12）约 0.011–0.013。大多数
   候选在统计上不可区分，中游候选的 beam 选择近乎随机。
3. **逐任务的涨跌巨大且互相抵消。** v2 运行里 best 对种子的逐任务差值
   在 −0.40…+0.65 之间（26 条涨超 0.2，12 条跌超 0.2）；+0.049 的 val
   均值提升只是巨大反向波动的微小残差，主导因素是生成/judge 方差。
4. **搜索一轮就饱和。** 两次运行的 4 个第 2 轮子候选全部低于父候选。这个
   配置下加轮数不会有用；第一次批评已经摘完了低垂果实（批评文本几乎全部
   针对规则合规——字数上限、恰好五键、person 替代性别词、不提 meta 词——
   所有 gradient 里"gate"只出现 1 次）。
5. **硬门在 val 上确实有反应**（种子有 4 条 val 任务低于 0.3，best 为
   0 条），但硬门触发主要由视频内容驱动，效应小、且不能干净地迁移到 test。

### 证据附录（诊断背后的具体数字）

以下数字全部由 `results/<run_id>/report.json`（每个候选的 100 条 val
逐任务 reward 向量）及其内嵌的 gradient 批评文本计算得出。

**A. 各候选的 val 分数（每次运行 9 个候选）。**

| 候选 | v1 运行（父候选） | v2 运行（父候选） |
| --- | --- | --- |
| v0 种子 | 0.764 | 0.594 |
| v1（R1，v0） | 0.788 | 0.637 |
| v2（R1，v0） | 0.774 | 0.615 |
| v3（R1，v0） | 0.781 | **0.643** |
| v4（R1，v0） | 0.773 | 0.615 |
| v5（R2） | 0.746（← v3） | 0.613（← v1） |
| v6（R2） | 0.740（← v3） | 0.617（← v1） |
| v7（R2） | 0.775（← v1） | 0.627（← v3） |
| v8（R2） | 0.784（← v1） | 0.617（← v3） |

**两次运行的所有第 2 轮子候选都低于其父候选**（8 个全部退化）——这就是
"搜索一轮后饱和"的依据。

**B. 噪声底 vs 候选间差距。** 逐任务 reward 的平均 SD：0.113（v1 运行）/
0.133（v2 运行）。val n=100 时单候选均值的 SE 为 0.0113 / 0.0133。候选
均值跨度（max−min）：0.057（v1 运行）/ 0.049（v2 运行）——整个候选群只
横跨约 4 个 SE，中游候选相差不到 1–2 个 SE，它们之间的 beam 选择由噪声
主导。

**C. best 对种子的逐任务涨跌（同一 val split）。**

| | v1 运行（v1−v0） | v2 运行（v3−v0） |
| --- | --- | --- |
| 平均差值 | +0.034 | +0.049 |
| 差值分位数（min/q1/中位/q3/max） | −0.42 / −0.07 / +0.03 / +0.14 / +0.50 | −0.40 / −0.09 / +0.03 / +0.20 / +0.65 |
| 涨幅 > 0.2 的任务数 | 13 | 26 |
| 跌幅 > 0.2 的任务数 | 10 | 12 |

均值上的净提升只是巨大反向波动的微小残差——单条任务的表现由生成/judge
方差主导。

**D. 硬门在 val 上的反应（v2 运行）。** reward < 0.3（触发硬门或失败）的
val 任务数：种子 v0 = 4 条，best v3 = 0 条。

**E. 批评文本实际在谈什么。** v2 运行所有 gradient 批评文本的关键词计数：
"rule" 45 次、"courier" 30 次、"scene" 29 次、"compliance" 9 次、
"judge" 2 次、"gate" 1 次。critic 给出的可操作建议压倒性地指向确定性的
规则合规部分（字数上限、恰好五键、person 替代性别词、不提 meta 词）——
与下面 F 的跨刻度结果一致：被改善的正是规则合规。

**F. 跨刻度迁移。** 只在 v1 下训练的 `tuned_v1` 在 v2 刻度上 +0.031、在
自己的 v1 刻度上却 −0.004；`tuned_v2` 分别为 +0.035 / +0.006。两个 prompt
改善的是同一样东西——只有 v2 显式度量的那部分（规则/结构）。如果 tuned
prompt 真的改善了语义描述质量，v1 刻度（0.6 judge 权重）也应该动。它没动。

### 后续动作（按优先级）

1. **先分解再继续优化**：从 `results/eval_*.json` 的组件明细统计 baseline
   vs tuned 的 rule_compliance、分字段 judge 均值和硬门触发率，量化还剩
   多少 prompt 可修复的余量（很可能已不多）vs 感知能力封顶的余量。
2. **若 judge_detail 是天花板，改输入而不是改 prompt**：更多/更密的帧、
   更高分辨率、或更强的多模态部署（`AZURE_OPENAI_DEPLOYMENT`）。prompt
   调优无法让模型看见帧里没有的东西。
3. **降噪让 APO 能爬更细的梯度**：`reward/v2/config.yaml` 设
   `judge_samples: 3`；考虑缓存生成结果离线重打分，把 reward 效应与生成
   方差分离。
4. **改 beam 形状而不是加深**：两次运行第 2 轮都在退化，预算应花在第一轮
   多样性上——如 `--branch-factor 4..6 --beam-rounds 2`——并加大
   `--gradient-batch-size` 让批评看到更多失败模式（包括触发硬门的任务）。
5. **人工复核 test 上仍触发硬门的任务**：若 scene/courier 在帧里确实
   有歧义，那是要与客户讨论的数据/标注问题，不是 prompt 问题。
6. **Test split 太小（30 条）**——仅均值 SE 就有约 0.02–0.03；要分辨 0.01
   量级的效应需要更大的 test split（见
   [dataset-sizing.zh.md](dataset-sizing.zh.md)）。
