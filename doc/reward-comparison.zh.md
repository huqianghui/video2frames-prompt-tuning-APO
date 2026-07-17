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

## 5. 端到端对比结果（待回填）

_两次 APO 训练与 2×2 评估完成后在此记录：run ID、beam 参数、各单元格
mean_reward、相对 baseline 的提升、结论。_
