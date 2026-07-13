# APO 元 Prompt 定制（POML）

[English](apo-poml-customization.md) | **中文**

APO 本身也由两个 prompt 驱动——作用于"被调优 prompt"之上的*元 prompt*。本文
说明它们各自的作用、为什么框架默认版对本项目不够用，以及
[`prompts/`](../prompts) 下项目定制版到底改了什么。

## 1. 这两个文件的作用

APO beam search 的每一轮扩展都分两步 LLM 调用完成
（`agentlightning/algorithm/apo/apo.py` 中的 `textual_gradient_and_apply_edit`）：

| 步骤 | 模板 | 模型（环境变量） | 输入 | 输出 |
| --- | --- | --- | --- | --- |
| 文本梯度 | `text_gradient_*.poml` | `APO_GRADIENT_MODEL`（默认 `gpt-4.1`） | 当前 prompt + 一批 rollout trace（消息、reward） | **批评**：低分的具体原因和可验证的修改建议（bullet list） |
| 应用编辑 | `apply_edit_*.poml` | `APO_APPLY_EDIT_MODEL`（默认 `gpt-4.1-mini`） | 当前 prompt + 批评 | **改写后的 prompt**，作为下一个候选 |

批评就是"梯度"，编辑就是"更新步"——这两个元 prompt 的质量直接决定搜索的
上限。

框架默认行为是每次扩展从 3 个 gradient 变体、2 个 edit 变体中*随机*选一个
（`agentlightning/algorithm/apo/prompts/*.poml`）。像本项目这样每步只指定一个
文件，还能让运行更可复现。

## 2. 为什么默认版在这里不够用

默认模板与任务无关：只说"提高 reward"，让梯度模型自己从 trace 里推断目标。
对本项目这留下三个真实的失效模式：

1. **编辑模型可能悄悄弄丢 JSON 契约。** 我们的 reward 对任何"不是恰好含 5 个
   字段的合法 JSON 对象"的输出直接记 0。一次"简化格式说明"的改写会让该候选
   的所有 rollout 都得 0 分——整个扩展的预算白白浪费。
2. **梯度模型不知道 reward 的结构。** 它看不出三个文本字段占 0.6 权重、两个
   分类字段各占 0.2，批评可能追逐低价值的修改。它也不知道内容安全过滤器的
   拒绝只取决于输入帧——不告诉它，它就会把 prompt 解决不了的失败归咎于
   prompt。
3. **编辑模型可能自作主张加媒体占位符。** 本项目中 `<frame n | Xs>` 占位符段
   和图片是 agent 在运行时追加在被调优指令*之后*的；改写若重新引入 `<video>`
   或帧标记，会与该段重复或冲突。

有一类其他 APO 项目需要的定制在这里被**刻意省略**：花括号/占位符保护。示例
里的 strict 模板禁止字面 `{`/`}`，是因为它们的被调优 prompt 要经过 Python
`str.format` 渲染。我们的 agent 原文直传模板文本
（`frame_agent.py`：`fixed_prompt = prompt_template.template`），prompt 里出现
带花括号的 JSON 示例是安全的——并且被明确允许。

## 3. 相对框架默认版的最终改动

两个文件都以 `*_variant01.poml` 为底，只改下表所列内容：

| 文件 | 改动 | 目的 |
| --- | --- | --- |
| `prompts/text_gradient_video2frames.poml` | 新增 **Optimization Objective** 一节：5 字段合法 JSON 契约（否则 0 分）、0.2/0.2/0.6 reward 公式与 judge 标准、"内容过滤器拒绝与 prompt 无关" | 让批评瞄准真正影响 reward 的方向 |
| | 新增 **Critique Constraints** 一节：不许建议加 `<video>`/帧占位符（运行时追加）、不许改动 5 个字段名、允许带花括号的 JSON 示例 | 把批评限制在任务契约之内 |
| `prompts/apply_edit_video2frames.poml` | 删掉 "Preserve placeholder variables inside curly brackets" | 我们的模板没有占位符，这条在此只会误导 |
| | 新增三条改写规则：必须保留 5 字段合法 JSON 要求、不许加 `<video>`/帧占位符、允许花括号 JSON 示例 | 防止改写破坏 reward 契约 |
| | output-format 改为"只返回改进后的 prompt 正文" | 去掉占位符相关措辞 |

其余部分——experiment 循环、算法填充的 `{{ prompt_template }}` /
`{{ critique }}` / `{{ experiments }}` 槽位——与默认版完全一致，因此这两个文件
可直接用于 `APO(gradient_prompt_files=..., apply_edit_prompt_files=...)`。

## 4. 使用方式

`apo_train.py` **默认**使用项目定制模板：

```bash
.venv/bin/python apo_train.py                 # 项目元 prompt（prompts/）
.venv/bin/python apo_train.py --default-poml  # 框架内置模板
```

`results/summary.json` 会记录本次用的是哪套模板（`"custom_poml"` 字段），方便
两套模板之间做可追溯的 A/B 对比。`tests/test_apo_train.py` 中的离线测试守护
模板必需的槽位和契约关键词。

如果与客户沟通后 reward 发生变化（见 [reward-design.zh.md](reward-design.zh.md)），
例如调整权重或改成按字段打分，请同步更新 gradient 模板的
**Optimization Objective** 一节——否则搜索会按过期的 reward 描述做优化。
