# Video2Frames Prompt 调优（APO）

[English](README-en.md) | **中文**

使用 [Agent-Lightning](https://github.com/huqianghui/agent-lightning) 的 APO
（Automatic Prompt Optimization，自动 Prompt 优化）算法，调优视频监控帧分析
任务中固定不变的指令 prompt。

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

本项目必须运行在**从 [huqianghui/agent-lightning](https://github.com/huqianghui/agent-lightning)
fork 源码构建的 agent-lightning 0.3.1** 上（PyPI 的 0.3.0 版本缺少所需功能，
且该 fork 含额外的日志与测试改动）。`requirements.txt` 已按 commit 固定该 fork：

```bash
git clone https://github.com/huqianghui/video2frames-prompt-tuning-APO.git
cd video2frames-prompt-tuning-APO
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

> **从这里开始：** 运行任何命令之前，先读
> [doc/optimization-stages.md](doc/optimization-stages.md)，定位自己
> 处于哪个优化阶段（测量 / 模型 / prompt / 数据）、该拉哪个杠杆——下面的
> 命令是*操作*，那份指南是*策略*。

```bash
# 1. 准备数据集（分层采样；从 Azure 解析帧 blob）。
#    采样与切分按 (family, is_courier_action) 联合分层，每个 split 都还原
#    候选池的标签比例；val split 保证 courier 正例比例不低于
#    --val-courier-min（默认 0.15），并在日志中打印各 split 的
#    courier/scene_type/family 分布。scene_type 不设配额（只做分布报告）。
#    --probe-content-filter 会在采样时把每个候选视频送 Azure 内容安全过滤器
#    探测（约 3% 的视频无论 prompt 如何都会被拒），被 block 的自动顺延补采，
#    保证各 split 达到目标大小且全部任务可通过。探测结果按视频缓存在
#    data/content_filter_cache.json 中，重复运行本脚本（或
#    probe_content_filter.py）不会重复探测已探测过的视频。
.venv/bin/python prepare_data.py --train-size 40 --val-size 24 --test-size 30 --seed 42 --probe-content-filter

# 1b.（第二轮）扩大 train/val 时冻结 held-out 测试集：--freeze-test 保持
#     test.jsonl 原样不动，并把其中的视频从重采样中排除（--test-size 被忽略）。
#     注意：冻结模式下即使 seed 相同也不会复现上一轮的 train/val
#     （候选池已经变化）。
.venv/bin/python prepare_data.py --train-size 80 --val-size 100 --freeze-test --probe-content-filter

# 2.（可选）对已有 split 做内容安全过滤器审计。
.venv/bin/python probe_content_filter.py          # 只出报告（data/content_filter_probe.json）
.venv/bin/python probe_content_filter.py --apply  # 报告 + 删除被 block 的任务（不补采）
.venv/bin/python probe_content_filter.py --from-report  # 复用已有报告重新应用

# 3. 用 baseline prompt 调试单条 rollout。
.venv/bin/python frame_agent.py --limit 1

# 4. 端到端冒烟测试 APO 闭环（最小 beam，成本低）。
.venv/bin/python apo_train.py --smoke

# 5. 完整 APO 训练。每次运行分配一个时间戳 run ID：APO 日志写入
#    log/apo_<run_id>.log，所有产物写入 results/<run_id>/ —— 最佳 prompt
#    （best_prompt.txt）、完整优化报告（每轮候选 prompt、reward、gradient
#    批评、验证分数，report.md + report.json）、精简的 prompt 版本树
#    （派生关系、分数、胜出版本，tree.md），以及记录运行参数和 data/ 各
#    split 指纹（行数 + 哈希）的 summary.json。若 best prompt 赢过种子，
#    diffs.md 会给出其派生链每一步的 unified diff（如 v0 → v4 → v7）以及
#    种子 → best 的整体 diff。多次运行互不覆盖；results/latest 始终指向
#    最新一次运行。
.venv/bin/python apo_train.py

# 5b.（可选）从任意一次历史运行的日志重新生成报告。
.venv/bin/python generate_report.py --log log/apo_<run_id>.log --output-dir results/<run_id>

# 5c.（可选）只用已有的 report.md 生成版本树——不需要日志（例如从其他机器
#     拷贝来的 report.md）。此模式下没有 beam 存活标记。
.venv/bin/python generate_report.py --from-report results/latest/report.md

# 6. 在 held-out 测试集上对比 baseline 与调优后的 prompt。
.venv/bin/python evaluate.py --name baseline
.venv/bin/python evaluate.py --prompt results/latest/best_prompt.txt --name tuned

# 7.（可选）对比 reward 版本：同一个 prompt 分别用 v1、v2 打分，
#    再按任务逐条 join 两份结果。
.venv/bin/python evaluate.py --name baseline_v1 --reward-version v1
.venv/bin/python evaluate.py --name baseline_v2 --reward-version v2
.venv/bin/python compare_rewards.py results/eval_baseline_v1.json results/eval_baseline_v2.json
```

AgentOps SaaS 上传**默认关闭**（span 仍在本地采集，APO 所需的数据不受影响）。
只有想在 app.agentops.ai 上查看 session replay 时，才给 `apo_train.py` 加
`--enable-agentops-service`。

默认的 split 大小（40/24/30）是试点配置。如何根据目标效应量估算真正需要的
split 大小，以及数据集与 beam 超参数同步阶梯式扩容的操作手册，见
[doc/dataset-sizing.md](doc/dataset-sizing.md)。

## Test split 的作用与使用条件

三个 split 在 APO 中的分工不同，**test 不参与任何优化决策**：

| Split | 作用 | 谁在消费 |
| --- | --- | --- |
| train | 供 critic 计算文本梯度（`--gradient-batch-size` 条 rollout/次） | APO 梯度阶段 |
| val | 给每轮候选 prompt 打分、决定 beam 去留 | APO 选择阶段 |
| test | held-out 最终验收，只回答"调优后比 baseline 好多少" | `evaluate.py`（人工触发） |

**使用条件（跑 test 前逐条确认）：**

1. **APO 真正产出了赢过种子的 prompt**——查看 `results/<run_id>/report.md` 或日志末尾：若 best prompt 仍是 v0（`Best prompt not updated`），跑 test 没有意义，先回去改数据/评估配置重跑 APO。
2. **确认所评估的 `best_prompt.txt` 来自正式运行而非 smoke**——每次运行（包括 smoke）都会写入独立的 `results/<run_id>/` 目录，而 `results/latest` 指向最近一次运行，它可能是 smoke。检查该运行目录下 `summary.json` 里的 beam 参数（smoke 为 1/1/1），确认拿到的是正式运行的产物。
3. **test 必须保持"未见过"**——优化迭代期间不要反复在 test 上试分数；调参、选 prompt 一律只看 val。test 每被用于一次决策，最终数字的可信度就打一次折扣。一轮优化收敛后跑一次即可。

**步骤与产出：**

```bash
.venv/bin/python evaluate.py --name baseline                                       # baseline prompt
.venv/bin/python evaluate.py --prompt results/latest/best_prompt.txt --name tuned  # 调优后 prompt
```

两次运行分别写入 `results/eval_baseline.json` 与 `results/eval_tuned.json`（含 mean_reward 和 per-task 明细），对比 `mean_reward` 即最终结论。若差距小于 val 上观测到的评估噪声（试点配置下约 ±0.015），不要宣称有提升——先按 [doc/dataset-sizing.md](doc/dataset-sizing.md) 扩大 split 再验证。

## Reward

reward 是本项目迭代最频繁的部分，因此被抽取成独立的版本化包：`reward/`
下每个版本一个子目录（`reward/v1/`、`reward/v2/`、……），各自包含实现
（`reward.py`）、可调参数（`config.yaml`），以及可选的 APO 元 prompt 覆盖
（`*.poml`）。所有版本实现同一个 `RewardFunction` 接口（`reward/base.py`），
agent、APO 训练与评估对版本完全无感。

**版本选择：** 给 `frame_agent.py` / `apo_train.py` / `evaluate.py` 传
`--reward-version vN`，或设置环境变量 `REWARD_VERSION`（显式参数 > 环境
变量 > 默认 `v1`）。`apo_train.py` 会把解析后的版本固定到环境变量，保证
fork 出的 runner 进程用同一个 reward 打分，并把版本号与完整 reward 配置记录
到该次运行的 `summary.json`。

**v1（默认）**——`[0, 1]` 区间的混合 reward：

- `0.2` × `scene_type` 精确匹配
- `0.2` × `is_courier_action` 精确匹配
- `0.6` × LLM judge 对 `english_detail` / `brief` / `title` 的语义评分

**v2（hybrid reward 升级版）**——分字段 judge + 机械规则合规 + 乘性硬门
（按 SkillOpt-04 分析文章重设计）：

- 软分：`0.45 × judge_detail + 0.20 × judge_brief + 0.10 × judge_title +
  0.25 × rule_compliance`，其中 `rule_compliance` 是一组确定性 0/1 检查的
  均值，检查项直接来自 baseline prompt 的硬性风格规则（恰好 5 个 JSON 键、
  字数上限、用 "person" 而非性别词、不提 camera/frame/timestamp、
  Non-Notable 触发前缀一致性）。
- 硬门以乘法作用于软分：场景判错 × 0.5，courier 误报 × 0.3，courier
  **漏报** × 0.2（漏掉真实快递员是最贵的错误；这一不对称假设需与客户确认）。
- 可选的 judge 降噪：在 `reward/v2/config.yaml` 中设 `judge_samples: 3`
  （并把 `judge_temperature` 调为非零）即对多次 judge 调用取中位数。
- `reward/v2/text_gradient_video2frames.poml` 向 APO 优化器描述 v2 的目标
  函数（每个版本自带自己的 text-gradient 元 prompt；见下文 "APO 元 Prompt"）。

两个版本对非法 JSON 输出都打 `0` 分。被 Azure OpenAI 内容安全过滤器拒绝的
请求同样得 `0` 分——拒绝只取决于输入帧，对每个候选 prompt 都完全相同。对
默认 94 条样本的探测发现约 3% 被 block（不只 `ucf_crime`；部分 `Charades`
视频也会触发）。用 `prepare_data.py --probe-content-filter` 采样可预先排除
被 block 的视频（reward=0 的兜底仍覆盖漏网的情况），或用
`probe_content_filter.py` 审计已有 split。

**版本对比：** 用不同的 `--name` 标签把同一个 prompt 分别在两个 reward 下
评估，再按任务 join 两份结果：

```bash
.venv/bin/python evaluate.py --name baseline_v1 --reward-version v1
.venv/bin/python evaluate.py --name baseline_v2 --reward-version v2
.venv/bin/python compare_rewards.py results/eval_baseline_v1.json results/eval_baseline_v2.json
```

`compare_rewards.py` 打印逐任务 delta 表、按 family 与整体的均值，并把
join 后的结果写到输入文件旁。实测结果（baseline 刻度校准、解读与 2×2
端到端对比计划）记录在
[doc/reward-comparison.md](doc/reward-comparison.md)。

**新增版本：** 把现有目录复制为 `reward/v3/`，改 `config.yaml` 和
`reward.py`（保持 `RewardFunction` 接口和 `get_reward()` 工厂），并把版本
目录里的 `text_gradient_*.poml` 改写成新目标函数（在 `apo_meta_prompts`
中声明）——`--reward-version v3` 即可自动使用。

v1 的设计理由、大规模训练前需要与客户确认的假设，见
[doc/reward-design.md](doc/reward-design.md)；v2 的设计遵循 SkillOpt-04
文章（[SkillOpt 系列 04](https://github.com/huqianghui/mindforge/blob/main/Notes/AI/SkillOpt/SkillOpt%E7%B3%BB%E5%88%9704%EF%BC%9AAPO%C3%97SkillOpt%E8%81%94%E5%90%88%E5%B1%95%E6%9C%9B%E2%80%94%E2%80%94%E5%85%88%E6%8E%A2%E7%B4%A2%E5%90%8E%E7%B2%BE%E4%BF%AE%E7%9A%84%E4%B8%A4%E6%AE%B5%E5%BC%8F%E7%AE%A1%E9%81%93%E4%B8%8E%E9%80%89%E5%9E%8B%E7%AE%97%E8%B4%A6%E6%96%B9%E6%B3%95.md)）。

## APO 元 Prompt

APO 本身由两个元 prompt 驱动：*text gradient* 模板根据 rollout trace 批评当前
prompt，*apply edit* 模板据此改写。两者都在 reward 版本的 `config.yaml` 的
`apo_meta_prompts` 段显式声明（`null` = 用 `prompts/` 下的共享默认文件，
写文件名 = 用版本目录下的客户化文件），并按与 reward 的耦合程度划分归属：

- **text gradient** 描述优化目标（即 reward 公式），因此**归 reward 版本
  所有**——每个版本必须声明一份（`reward/v1/text_gradient_video2frames.poml`
  描述 0.2/0.2/0.6 目标，`reward/v2/...` 描述分字段 + 硬门目标）；版本
  没有声明时 `apo_train.py` 直接报错拒跑。
- **apply edit** 只编码与 reward 无关的不变量（5 字段 JSON 契约、禁止加入
  帧/`<video>` 占位符），因此共享的 `prompts/apply_edit_video2frames.poml`
  是所有版本的默认文件。

加 `--default-poml` 可回退到框架内置模板。这两个文件的作用与相对默认版的
具体改动，见 [doc/apo-poml-customization.md](doc/apo-poml-customization.md)。

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

beam 超参数（`--beam-rounds` / `--beam-width` / `--branch-factor`）的含义与
调参方法、并发模型、运行时长与 Azure OpenAI quota 的估算公式、`--n-runners`
和批大小的选择方法（如大 VM 上用 `--n-runners 12 --gradient-batch-size 8`），
见 [doc/performance-tuning.md](doc/performance-tuning.md)。

> **注意：** Python 3.14 把 Linux 的默认启动方式改为 `forkserver`（同样需要
> pickle 入口函数），因此那些环境也会自动回退到串行模式。在上游修复之前
> （microsoft/agent-lightning：进程入口改用模块级函数、活动 tracer 改用
> thread-local/contextvar），并行运行请锁定 Python ≤ 3.13。

## Dashboard（可选）

在 Linux 上运行时，日志中可能出现：

```
ERROR    Dashboard directory not found at .../agentlightning/dashboard
```

**这个报错无害**——dashboard 是一个可选的 Web 界面，用于浏览 store 中的数据
（rollouts、spans、traces），没有它训练照常进行。报错的原因是本项目从源码安装
agent-lightning，而前端尚未构建。如需启用 UI，构建一次即可
（`cd <agent-lightning>/dashboard && npm install && npm run build`）然后重启。
详见 [doc/dashboard.md](doc/dashboard.md)，其中也解释了为什么
macOS/Windows 的 shm 回退模式下根本没有 dashboard。

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
| `prepare_data.py` | 把 pandas 导出转换为 `data/{train,val,test}.jsonl` 和 `data/baseline_prompt.txt`。按 (family, `is_courier_action`) 联合分层，val 有 courier 正例比例下限（`--val-courier-min`）；`--freeze-test` 在不动 test split 的前提下重建 train/val；`--probe-content-filter` 跳过被内容安全过滤器 block 的视频。 |
| `probe_content_filter.py` | 把任务送内容安全过滤器探测；结果按视频缓存在 `data/content_filter_cache.json`，按 split 报告 block 比例，可选删除被 block 的任务。 |
| `frame_agent.py` | `@rollout` 帧分析 agent、帧占位符构建、调试 CLI；打分委托给版本化的 `reward/` 包。 |
| `reward/` | 版本化 reward 包：`base.py`（共享 `RewardFunction` 接口、JSON 解析、judge 工具），`v1/`（0.2/0.2/0.6 混合 reward），`v2/`（分字段 judge + 规则合规 + 硬门）。每个版本有 `reward.py` + `config.yaml` + 自己的 text-gradient POML。用 `--reward-version` / `REWARD_VERSION` 选择。 |
| `compare_rewards.py` | 按任务 join 两份 `evaluate.py` 结果（如同一个 prompt 在 reward v1 与 v2 下的得分），打印逐任务 delta 与按 family/整体的均值。 |
| `apo_train.py` | APO 训练入口；每次运行写 `log/apo_<run_id>.log` 和 `results/<run_id>/`（`best_prompt.txt`、带数据指纹的 `summary.json`、运行报告），并更新 `results/latest` 指向。元 prompt 按 reward 版本的 `apo_meta_prompts` 配置解析（`--default-poml` 回退框架模板）。 |
| `prompts/apply_edit_video2frames.poml` | 共享的、与 reward 无关的 APO apply-edit 元 prompt（5 字段 JSON 契约、禁止帧占位符）。reward 相关的 text-gradient 元 prompt 在各 `reward/<version>/` 目录下。 |
| `generate_report.py` | 把 APO 运行日志（`--log log/apo_<run_id>.log`）解析为 `report.md` / `report.json`（每轮候选 prompt、reward、gradient 批评）、`tree.md`（精简版本树：派生关系、分数、beam 存活、胜出版本），以及（best prompt 赢过种子时）`diffs.md`（派生链每步 diff + 种子 → best 整体 diff），写入 `--output-dir`。 |
| `evaluate.py` | 在指定数据集 split 上评估一个 prompt 文件；写 `results/eval_<name>.json`（记录 reward 版本；用 `--reward-version` 选择）。 |
| `doc/en-us/optimization-stages.md` / `doc/optimization-stages.md` | **入门策略指南**：模型/数据/prompt 三个杠杆的定位、如何从 reward 组件判断当前所处的优化阶段，以及逐阶段路线图与退出/终止条件（英/中）。 |
| `doc/en-us/dataset-sizing.md` / `doc/dataset-sizing.md` | 数据集规模选择指南（噪声/SE 计算）、阶梯式扩容与 beam 超参调优手册（英/中）。 |
| `doc/en-us/reward-design.md` / `doc/reward-design.md` | Reward 定义、设计理由与待客户确认的问题清单（英/中）。 |
| `doc/en-us/reward-comparison.md` / `doc/reward-comparison.md` | v1 vs v2 实测对比记录：test split 上的 baseline 刻度校准、结果解读，以及 2×2 端到端对比操作手册（英/中）。 |
| `doc/en-us/apo-poml-customization.md` / `doc/apo-poml-customization.md` | APO 元 prompt 的作用、定制原因与相对框架默认版的具体改动（英/中）。 |
| `doc/en-us/dashboard.md` / `doc/dashboard.md` | Agent-Lightning dashboard 是什么、为何 "Dashboard directory not found" 报错无害、如何构建与访问 UI（英/中）。 |
| `doc/en-us/performance-tuning.md` / `doc/performance-tuning.md` | Beam 超参数（rounds/width/branch-factor）的意义与调参决策表、并发模型、运行时长/quota 估算公式、`--n-runners` 与批大小的选择方法（英/中）。 |
| `README-en.md` / `README.md` | 本文档（英/中）。 |
| `tests/` | 离线单元测试（仅 fixture，无客户数据、无网络）。 |
| `conftest.py` | 让 `tests/` 可以 import 项目模块。 |
| `requirements.txt` | 从源码安装 agent-lightning 0.3.1（`-e ..[apo]`）及项目依赖。 |
| `pyrightconfig.json` | 把 pyright 指向项目 virtualenv。 |
| `.gitignore` | 让客户数据、生成数据集、日志、结果和 env 文件不入 git（文件夹通过 `.gitkeep` 保留）。 |
