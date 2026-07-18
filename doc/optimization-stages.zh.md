# APO 优化阶段指南：模型、数据、Prompt 三个杠杆怎么排序

[English](optimization-stages.md) | **中文**

新接手或恢复这个项目时，最重要的问题不是"怎么调 prompt"，而是**"当前处于
哪个阶段、该拉哪个杠杆"**。本文把 2026-07 两次端到端实验
（[reward-comparison.zh.md](reward-comparison.zh.md) 第 5–6 节）的结论固化成
一个可操作的阶段路线图：先定位、再行动，避免在错的阶段用错的杠杆。

## 0. 核心结论：三个杠杆不在同一个维度上

| 杠杆 | 能动的分数 | 特性 | 实测证据 |
| --- | --- | --- | --- |
| **换 target 模型** | +0.04 起步，**唯一能抬天花板的** | 收益来自感知能力（judge_detail 那 0.45 权重），prompt 够不到 | gpt-5.4 零调优 +0.041；未调优 prompt 已超旧模型的调优 prompt |
| **APO 调 prompt** | +0.03–0.05，**一次性红利** | 只能收割确定性的规则合规切片（0.25 权重），一轮编辑摘完就饱和 | 两次 run 全部收益来自第一轮；8/8 个第二轮子候选退化 |
| **Data size** | +0，但决定**能不能看见前两者** | 不产生收益，产生测量分辨率 | test=30 的配对 SE 0.035 ≥ 要测的效应 0.03–0.04，结论永远"尚不足以下定论" |

一句话：**模型抬天花板、prompt 收一次性红利、数据定尺子精度。**

**"感知能力"的确切含义**：模型确实收到了帧图片，但从抽样的帧里*辨认不出*
发生了什么（动作太细微、帧间信息丢失、物体太小），导致 english_detail 漏报
或误报事件——描述真的错了，不是评分假象。指令能改变*怎么说*，改变不了
*看见了什么*，所以这块差距 prompt 永远够不到，杠杆是换模型或加密帧。

## 1. 先定位：你在哪个阶段

不用猜，跑一次 baseline 评估、分解 reward 组件，每种短板对应一个杠杆：

```bash
.venv/bin/python evaluate.py --name baseline --reward-version v2
# 组件明细在 results/eval_baseline.json 及运行日志中
```

| 观察到的症状 | 瓶颈 | 该拉的杠杆 | 对应阶段 |
| --- | --- | --- | --- |
| 候选/版本之间的分差 < 2×SE，什么结论都下不了 | 尺子 | 扩 test、`judge_samples`、重复评估 | 阶段 1 |
| `judge_detail`/`brief`/`title` 低，rule_compliance 不低 | 感知 | 换更强的多模态模型、加密帧 | 阶段 2 |
| `rule_compliance` 低（字数超限、缺键、性别词、meta 词） | prompt | 跑一次 APO | 阶段 3 |
| 硬门（scene/courier）触发率高且帧里确实有歧义 | 数据/标注 | 人工复核、与客户对齐 | 阶段 4 |

判断"是感知问题而不是 prompt 没调好"的四条证据链（都可以在自己的运行上
复现）：

1. **APO 是否已饱和**：第二轮子候选是否全部低于父候选、critique 是否只在
   重复规则合规类问题（`results/<run_id>/report.json` 里有每个候选的
   gradient 文本）；
2. **跨刻度反证**：调优后的 prompt 在 v2 刻度提升、但 v1 刻度（0.6 权重是
   语义 judge）不动 → prompt 改善的只是格式/规则，语义那块碰不着；
3. **控制变量探测**：同一 baseline prompt、同一 test split、同一 judge，
   只换 `AZURE_OPENAI_DEPLOYMENT`——分数动了就是感知，没动就不是；
4. **组件分解**：短板落在 judge_detail（感知）还是 rule_compliance
   （prompt）。

## 2. 阶段路线图

### 阶段 0 —— 冒烟（每个环境一次）

验证闭环能跑，不看分数。见
[dataset-sizing.zh.md](dataset-sizing.zh.md) 第 6 节 Stage 0。

### 阶段 1 —— 磨尺子：主要矛盾是测量，不是优化

不先做这步，后面任何改动都无法判断有没有效。**注意：不是重新平衡三个
split**——train 80 / val 100 已经够用（实测校准见
[dataset-sizing.zh.md](dataset-sizing.zh.md) 第 8 节），只动两件事：

1. **test 一次性扩到 ~100 条并冻结**（`prepare_data.py --test-size 100
   --freeze-test`）。副作用：所有旧 baseline 锚点作废，需重测一次 baseline。
2. **降低单次测量方差**：`reward/v2/config.yaml` 设 `judge_samples: 3`；
   关键对比同一 split 跑 2–3 遍取平均。

**退出条件**：test 上的配对 SE ≤ 你要分辨的效应量的一半（本项目效应量
~0.03–0.04 → 需要配对 SE ≤ 0.02，即 ~100 条 + 重复评估）。

### 阶段 2 —— 换引擎：当前分数段的主要矛盾是感知

用控制变量探测确认收益，再决定切换：

```bash
# 只换 AZURE_OPENAI_DEPLOYMENT，其余全部不变；至少跑 2–3 遍取平均
.venv/bin/python evaluate.py --name baseline_<newmodel>_v2 --reward-version v2
```

判读：配对差值的均值 ≥ 2×配对 SE 才算确认；同时查最大退化任务的组件明细
——如果新模型的失败属于啰嗦/规则类，那是 prompt 可修复的，阶段 3 反而有
额外空间。想探上限可单跑一次更大的模型档位（如 pro）做参照。

**退出条件**：新模型确认后切换 `AZURE_OPENAI_DEPLOYMENT`，进入阶段 3。

### 阶段 3 —— 在新模型上收一次 prompt 红利（只收一次）

新模型犯的是*不同类型*的错，规则合规切片会重新出现可修复空间。配置按已
验证的结论：**宽度换多样性、不加深**：

```bash
.venv/bin/python apo_train.py --reward-version v2 \
  --branch-factor 4 --beam-rounds 2 --gradient-batch-size 8 \
  --n-runners 12   # Linux 上按 performance-tuning.zh.md 选并发
```

**终止信号（重要）**：`report.json` 里第二轮子候选全部低于父候选、且
critique 开始重复同样的问题 → 搜索已饱和，**停止**。不要在同一个模型上
反复多轮调 prompt——候选间分差通常 ≤ 1–2 个 SE，那是在噪声里挑硬币。

**退出条件**：在冻结 test 上确认 tuned vs baseline 的配对差距 ≥ 2×SE，
或触发终止信号后接受当前 best。

### 阶段 4 —— 剩余差距归数据/输入

扩 test + 换模型 + 一次 APO 之后 judge_detail 仍是短板时：

- **先测 judge_detail 的天花板（几乎零成本）**：把 ground truth 的
  `english_detail` 自己当作"模型输出"送进 judge 打分——得到的就是
  （帧、标注、judge）这个组合的理论上限。客户标注通常是看*完整视频*
  写的，抽样帧里看不见的事件会封死分数上限，换多强的模型都够不到。
  如果天花板只有 0.8，当前分数其实离顶没那么远，"低"是刻度假象；
  只有*低于天花板*的那段缺口才值得追。
- **人工抽查 judge_detail 最低的任务**：`evaluate.py` 已把每个任务的
  组件明细写进 `results/eval_<name>.json`，按 `judge_detail` 排序取
  最低的 ~10 条，人眼对照帧、ground truth 和模型输出。三种结局分别
  指向不同的嫌疑人：人能从帧里看出来而模型没看出 → 感知问题（换更强
  模型，回阶段 2）；人从帧里也看不出但标注说发生了 → 帧密度/标注
  错位（改输入，或与客户对齐标注）；模型说得其实没错但分低 → judge
  校准问题（回阶段 1）。
- **改输入而不是改 prompt**：更密的帧采样、更高分辨率——模型看不见帧里
  没有的东西；改完回到阶段 2 的探测流程验证。
- **人工复核仍触发硬门的任务**：scene/courier 在帧里真有歧义的，是标注/
  需求问题，要与客户对齐，不是 prompt 问题。

## 3. 循环规则

- 换了 target 模型或输入（帧密度/分辨率）→ 回阶段 3 重收一次 prompt 红利；
- 换了 reward 定义或 judge 模型 → 尺子变了，回阶段 1 重测 baseline 锚点；
- 任何时候拿不准 → 回第 1 节的定位表，先分解组件再动手。

在错的阶段拉错的杠杆（最常见：感知瓶颈期反复调 prompt）只会花钱买噪声。
