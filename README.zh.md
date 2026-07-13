# Video2Frames Prompt 调优（APO）

[English](README.md) | **中文**

使用 [Agent-Lightning](../README.md) 的 APO（Automatic Prompt Optimization，自动
Prompt 优化）算法，调优视频监控帧分析任务中固定不变的指令 prompt。

客户数据集 `original_data/qwen_0318_swift_task.json` 包含 5850 条视频分析任务
（视频 → 结构化 JSON，字段为 `english_detail` / `brief` / `title` / `scene_type` /
`is_courier_action`）。本项目把每条视频任务改造成**多帧图片任务**：移除原始的
`<video>` 占位符，并在指令之后追加帧占位符段，每帧一个占位符：

```
<frame 1 | 0s> <frame 2 | 3s> ... <frame n | 3(n-1)s>
```

帧已预先抽取（约每 3 秒一帧；每个视频帧数不等）并存储在 Azure Blob Storage 中。
APO **只调优 prompt 中固定的指令部分**；帧占位符段由 agent 在运行时按任务重建。

> **重要：** `original_data/qwen_0318_swift_task.json` 是客户提供的数据，
> 绝不能 commit 或 push 到 GitHub。`original_data/`、`data/`、`log/`、`results/`
> 四个目录只以空文件夹形式入库（仅 `.gitkeep`）；其内容因包含或派生自客户数据而被
> git 忽略。请单独（如通过 scp）把 `original_data/`、`data/` 和仓库根目录的 `.env`
> 拷贝到训练机器上。

## 安装

本项目必须运行在**从本仓库源码构建的 agent-lightning 0.3.1** 上（PyPI 的 0.3.0
版本缺少所需功能）：

```bash
cd video2frames-prompt-tuning
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import agentlightning; print(agentlightning.__version__)"  # 应输出 0.3.1
```

## 配置

Blob 存储配置从仓库根目录的 `.env` 读取
（`blob4videodatasets_connection_string`、`blob4videodatasets_container_name`、
`blob4videodatasets_frames_folder_name`）。此外，还需在 `.env` 中补充（或 export）
以下 Azure OpenAI 变量：

| 变量 | 用途 | 默认值 |
| --- | --- | --- |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL | 必填 |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | 必填 |
| `OPENAI_API_VERSION` | Azure OpenAI API 版本（如 `2024-10-21`） | 必填 |
| `AZURE_OPENAI_DEPLOYMENT` | 接收帧图片做分析的多模态部署 | `gpt-4o` |
| `JUDGE_MODEL` | reward 中 LLM judge 使用的部署 | `gpt-4.1-mini` |
| `APO_GRADIENT_MODEL` | APO 用来批评 prompt 的部署 | `gpt-4.1` |
| `APO_APPLY_EDIT_MODEL` | APO 用来改写 prompt 的部署 | `gpt-4.1-mini` |
| `FRAMES_AS_BASE64` | 设为 `true` 时帧以 base64 data URI 发送（默认 SAS URL） | 不设置 |

## 工作流

```bash
# 1. 准备数据集（分层采样；从 Azure 解析帧 blob）。
#    --probe-content-filter 会在采样时把每个候选视频送 Azure 内容安全过滤器
#    探测（约 3% 的视频无论 prompt 如何都会被拒），被 block 的自动顺延补采，
#    保证各 split 达到目标大小且全部任务可通过。探测结果按视频缓存在
#    data/content_filter_cache.json 中，重复运行本脚本（或
#    probe_content_filter.py）不会重复探测已探测过的视频。
.venv/bin/python prepare_data.py --train-size 40 --val-size 24 --test-size 30 --seed 42 --probe-content-filter

# 2.（可选）对已有 split 做内容安全过滤器审计。
.venv/bin/python probe_content_filter.py          # 只出报告（data/content_filter_probe.json）
.venv/bin/python probe_content_filter.py --apply  # 报告 + 删除被 block 的任务（不补采）
.venv/bin/python probe_content_filter.py --from-report  # 复用已有报告重新应用

# 3. 用 baseline prompt 调试单条 rollout。
.venv/bin/python frame_agent.py --limit 1

# 4. 端到端冒烟测试 APO 闭环（最小 beam，成本低）。
.venv/bin/python apo_train.py --smoke

# 5. 完整 APO 训练。最佳 prompt 写入 results/best_prompt.txt，完整优化报告
#    （每轮候选 prompt、reward、gradient 批评、验证分数）写入
#    results/report.md + results/report.json。
.venv/bin/python apo_train.py

# 5b.（可选）从已有的 log/apo.log 重新生成报告，例如同一日志文件中更早的一次运行。
.venv/bin/python generate_report.py --run -1

# 6. 在 held-out 测试集上对比 baseline 与调优后的 prompt。
.venv/bin/python evaluate.py --name baseline
.venv/bin/python evaluate.py --prompt results/best_prompt.txt --name tuned
```

默认的 split 大小（40/24/30）是试点配置。如何根据目标效应量估算真正需要的
split 大小，以及数据集与 beam 超参数同步阶梯式扩容的操作手册，见
[dataset-sizing.zh.md](dataset-sizing.zh.md)。

## Reward

每条 rollout 用 `[0, 1]` 区间的混合 reward 打分：

- `0.2` × `scene_type` 精确匹配
- `0.2` × `is_courier_action` 精确匹配
- `0.6` × LLM judge 对 `english_detail` / `brief` / `title` 的语义评分
- 输出不是合法 JSON 对象 → 得 `0` 分。
- 被 Azure OpenAI 内容安全过滤器拒绝的请求得 `0` 分——拒绝只取决于输入帧，
  对每个候选 prompt 都完全相同。对默认 94 条样本的探测发现约 3% 被 block
  （不只 `ucf_crime`；部分 `Charades` 视频也会触发）。用
  `prepare_data.py --probe-content-filter` 采样可预先排除被 block 的视频
  （reward=0 的兜底仍覆盖漏网的情况），或用 `probe_content_filter.py` 审计
  已有 split。

## 执行策略与平台说明

`apo_train.py` 会自动选择执行策略（`execution_strategy()`）：

- **Linux + Python ≤ 3.13**（multiprocessing 启动方式为 `fork`，主要目标平台）：
  使用默认的 client/server 策略，并行 runner 进程——默认 `--n-runners 4`。
- **macOS / Windows**（启动方式为 `spawn`，或 Linux Python 3.14+ 的 `forkserver`）：
  回退到串行共享内存模式
  （`strategy={"type": "shm", "n_runners": 1, "main_thread": "algorithm"}`）
  并打印警告——冒烟测试和小规模运行没有问题。

设置回退是因为当前的 agent-lightning 运行时有两个平台相关的限制：

1. **默认 client/server 策略在 macOS 和 Windows 上会失败。**
   `ClientServerExecutionStrategy._spawn_runners`（`agentlightning/execution/client_server.py`）
   通过 `multiprocessing.get_context()`（即*平台默认*启动方式）启动 runner 进程，
   并把一个局部定义的闭包（`_runner_sync`）作为进程入口。Linux（Python ≤ 3.13）
   默认是 `fork`，从不 pickle 入口函数，一切正常。macOS 和 Windows 默认是 `spawn`，
   必须 pickle，于是报
   `AttributeError: Can't pickle local object 'ClientServerExecutionStrategy._spawn_runners.<locals>._runner_sync'`。
   Windows 没有 `fork`，无任何变通办法。

2. **共享内存模式无法并行 runner。**
   shm 策略在单进程内以线程运行 runner，但 tracer 注册的是*进程全局*的活动 tracer
   （`agentlightning/tracer/base.py` 中的 `set_active_tracer` 会抛出
   `An active tracer is already set`）。2 个以上 runner 线程时，所有时间上重叠的
   rollout 都会失败，所以 `n_runners` 必须保持 1。

**实际结论：** 大规模运行请使用 **Linux + Python ≤ 3.13**，`apo_train.py` 会自动
并行（用 `--n-runners` 调节）。macOS/Windows 上同一条命令也能正确运行，只是串行。

> **注意：** Python 3.14 把 Linux 的默认启动方式改为 `forkserver`（同样需要
> pickle 入口函数），因此那些环境也会自动回退到串行模式。在上游修复之前
> （microsoft/agent-lightning：进程入口改用模块级函数、活动 tracer 改用
> thread-local/contextvar），并行运行请锁定 Python ≤ 3.13。

## 冒烟测试

离线（无网络、无凭据）：

```bash
.venv/bin/pytest tests/ -v
```

在线（需要 blob 访问 + Azure OpenAI）：

```bash
.venv/bin/python prepare_data.py --train-size 2 --val-size 2 --test-size 2
.venv/bin/python frame_agent.py --limit 1
.venv/bin/python apo_train.py --smoke
```

## 文件清单

| 文件 | 作用 |
| --- | --- |
| `original_data/qwen_0318_swift_task.json` | 客户数据集（pandas `to_json` 导出）。**绝不入库。** |
| `blob_utils.py` | Azure Blob 工具：环境加载、视频→帧前缀映射、帧列举、SAS URL。 |
| `prepare_data.py` | 把 pandas 导出转换为 `data/{train,val,test}.jsonl` 和 `data/baseline_prompt.txt`；`--probe-content-filter` 跳过被内容安全过滤器 block 的视频。 |
| `probe_content_filter.py` | 把任务送内容安全过滤器探测；结果按视频缓存在 `data/content_filter_cache.json`，按 split 报告 block 比例，可选删除被 block 的任务。 |
| `frame_agent.py` | `@rollout` 帧分析 agent、帧占位符构建、混合 reward、调试 CLI。 |
| `apo_train.py` | APO 训练入口；写 `results/best_prompt.txt`、`results/summary.json` 和运行报告。 |
| `generate_report.py` | 把 `log/apo.log` 解析为 `results/report.md` / `report.json`（每轮候选 prompt、reward、gradient 批评）。 |
| `evaluate.py` | 在指定数据集 split 上评估一个 prompt 文件；写 `results/eval_<name>.json`。 |
| `dataset-sizing.md` / `dataset-sizing.zh.md` | 数据集规模选择指南（噪声/SE 计算）、阶梯式扩容与 beam 超参调优手册（英/中）。 |
| `README.md` / `README.zh.md` | 本文档（英/中）。 |
| `tests/` | 离线单元测试（仅 fixture，无客户数据、无网络）。 |
| `conftest.py` | 让 `tests/` 可以 import 项目模块。 |
| `requirements.txt` | 从源码安装 agent-lightning 0.3.1（`-e ..[apo]`）及项目依赖。 |
| `pyrightconfig.json` | 把 pyright 指向项目 virtualenv。 |
| `.gitignore` | 让客户数据、生成数据集、日志、结果和 env 文件不入 git（文件夹通过 `.gitkeep` 保留）。 |
