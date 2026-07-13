# Reward 设计与待客户确认的问题

[English](reward-design.md) | **中文**

reward 函数就是优化目标：APO 会朝 reward 更高的方向改写 prompt。如果 reward
编码了错误的优先级，APO 就会**精确地**优化错误的目标。本文记录当前 reward 的
定义、设计理由、其中哪些是只有客户才能确认的假设，以及在大规模训练之前如何
与客户展开这场对话。

## 1. 当前定义

实现在 `frame_agent.py`（权重常量在文件顶部，打分在 `compute_reward`，judge 在
`judge_text_fields`）：

```
reward = 0.2 × scene_type 精确匹配          （不区分大小写）
       + 0.2 × is_courier_action 精确匹配   （容忍 "true"/"false" 字符串）
       + 0.6 × LLM judge 对 english_detail / brief / title 的语义评分
```

两条硬性归零规则：

- 输出不是合法 JSON 对象 → `0` 分。
- 请求被 Azure 内容安全过滤器拒绝 → `0` 分（拒绝只取决于输入帧，对每个候选
  prompt 完全相同）。

judge（`JUDGE_MODEL`，默认 `gpt-4.1-mini`，`temperature=0`，结构化输出
`reason` + `score`）的指令是：判断生成文本与 ground truth 是否"描述相同的主体
和动作"，措辞可以不同，严格打分、允许部分得分。它对三个文本字段**合并输出一个
0–1 分**。

## 2. 为什么这么设计

1. **按字段性质分而治之。** 5 个输出字段中，`scene_type`（indoor/outdoor）和
   `is_courier_action`（bool）可客观判定——精确匹配零成本、零噪声、无歧义。
   三个自由文本字段不可能精确匹配，LLM judge 语义比对是唯一可行的评分方式。
2. **APO 需要连续的信号。** 只用精确匹配的话 reward 只有 5 个离散取值，梯度
   模型几乎无从批评。judge 的部分得分让"描述略有偏差"与"完全错误"可区分，
   这正是文本梯度步骤赖以工作的信息。
3. **权重跟随内容占比。** 三个文本字段是输出的主体（也是 prompt 最能影响的
   部分），给 `0.6`；两个分类字段各 `0.2`。
4. **归零规则排除与 prompt 无关的噪声。** 非法 JSON 意味着格式契约被破坏
   （下游无法消费输出——重罚）。内容过滤器的拒绝与候选 prompt 无关，计 0 分
   （并通过探针缓存在采样阶段就排除这些视频）可避免污染对比。

## 3. 只有客户能回答的问题

以下是内嵌在 reward 里的假设。假设错了，APO 就在精确地优化错误目标，所以要在
大规模训练**之前**确认：

| # | 问题 | 为什么重要 | 如果答案不同 |
| --- | --- | --- | --- |
| 1 | `is_courier_action` 是否是业务核心信号（这看起来像快递/配送检测产品）？误报和漏报的代价是否相同？ | 权重 0.2 意味着修好快递检测的 prompt 收益很小，APO 会转而优先文本质量。误分类代价通常是不对称的。 | 提高 `COURIER_WEIGHT`；把对称精确匹配换成非对称打分（如漏检快递员比误报代价更高）。 |
| 2 | 下游实际消费的是哪个文本字段——`brief`（展示给用户？）、`english_detail`（检索/存档？）还是 `title`？ | judge 目前只出一个合并分；一个改善了关键字段、但劣化了次要字段的 prompt 得分不变。 | 把 judge 拆成按字段打分、分别加权。 |
| 3 | ground truth 是怎么产生的——人工标注还是模型生成（数据集名暗示 SFT 蒸馏）？有无已知质量问题？ | judge 是*对照 GT* 打分的。GT 有噪声既压低可达上限，也可能把调优引向复现 GT 的毛病。 | 对 val/test 的子集做清洗或重标；或在 judge 指令里明确容忍特定的 GT 缺陷。 |
| 4 | 多大的提升值得上线（如平均 reward +0.05，或快递准确率 +X 个百分点）？ | 这就是 [dataset-sizing.zh.md](dataset-sizing.zh.md) 里的效应量 `δ`——决定 val/test 要多大、调优何时可以停。 | 训练前用规模公式重新确定 split 大小。 |
| 5 | 下游解析是严格 JSON，还是有容错（如剥 markdown 代码块）？ | 目前任何非 JSON 输出都记 0——最严厉的惩罚。 | 放宽解析 / 对可恢复的输出给部分分。 |
| 6 | 客户能否人工打分 10–20 条样例输出？ | 用于校准 LLM judge。如果 judge 分与人工判断不相关，必须*先*修 judge 评分标准再调优——judge 是整个系统的考官。 | 迭代 judge 的 prompt / 模型直到相关性可接受。 |

问题 1–4 应在花钱跑完整训练之前敲定；5–6 便宜，可并行核对。

## 4. 建议的下一步

1. **给客户发一份简报**（本文档即可用）：reward 公式、上面 6 个问题，外加
   2–3 条来自 `results/eval_baseline.json` 的真实打分样例，让讨论落在真实输出
   上而不是抽象概念上。
2. **并行跑试点**（[dataset-sizing.zh.md](dataset-sizing.zh.md) 的 Stage 1，
   默认 40/24/30）——它测出 reward 噪声 σ，也为第 1 步产出样例；即使之后权重
   变更，这一步也不浪费。
3. **把答案折回实现。** 权重变更只是三个常量（`frame_agent.py` 里的
   `SCENE_WEIGHT` / `COURIER_WEIGHT` / `JUDGE_WEIGHT`）；按字段打分或非对称
   快递打分也只是 `judge_text_fields` / `compute_reward` 的小范围局部修改，
   配套单测在 `tests/test_frame_agent.py`。
4. **然后才跑完整的 APO 阶梯**（Stage 2+）。大规模训练之后再改 reward 意味着
   重新付一遍训练成本——reward 这场对话是全项目最便宜的保险。
