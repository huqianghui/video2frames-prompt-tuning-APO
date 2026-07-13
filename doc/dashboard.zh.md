# Agent-Lightning Dashboard：它是什么，以及为什么 "Not Found" 报错无害

[English](dashboard.md) | **中文**

## 结论先行

如果训练日志中出现：

```
ERROR    Dashboard directory not found at /home/azureuser/agent-lightning/agentlightning/dashboard
```

**没有任何功能被破坏。** Dashboard 是一个*可选的* Web 界面。静态文件缺失时，
store server 只是跳过挂载 UI，所有 API 照常工作——数据准备、rollout、APO 训练、
评估完全不受影响。可以直接忽略这个报错，或按下文构建一次 dashboard 让它消失。

## Dashboard 是什么

Dashboard 是一个 Web 应用（React + Mantine UI），用于在实验运行中或结束后查看
`LightningStore` 里的内容：rollouts、attempts、spans/traces、resources 等。
它纯粹是一个**查看/调试界面**——不承载任何训练逻辑。

它由 `LightningStoreServer`（包装 store 的 HTTP 服务）托管。服务启动时
（`agentlightning/store/client_server.py` 的 `_setup_dashboard`）会在
**Python 包内**的 `agentlightning/dashboard/` 查找预构建的静态文件：

- 找到 → UI 挂载到服务根路径，日志打印
  `Agent-lightning dashboard will be available at <endpoint>`。
- 缺失 → 打印上面的 ERROR 后直接返回。`/v1/...` 的 store API 继续正常工作，
  只是浏览器 UI 不可用。

## 报错原因

有两个名字相近的目录：

| 路径 | 内容 |
| --- | --- |
| `<仓库>/dashboard/` | 前端**源码**（React/TypeScript，npm 工程） |
| `<仓库>/agentlightning/dashboard/` | 前端**构建产物**（由 `npm run build` 生成；`vite.config.mjs` 的 `outDir` 指向此处） |

PyPI 官方 wheel 包已内置构建产物，所以 `pip install agentlightning` 的用户
不会遇到这个错误。本项目是**从源码安装** agent-lightning
（`requirements.txt` 使用 `-e ..[apo]`），而构建产物不在 git 中——因此目录缺失。

## 构建 dashboard（可选）

需要 Node.js（建议 ≥ 18）：

```bash
cd /path/to/agent-lightning/dashboard
npm install
npm run build     # 静态文件输出到 ../agentlightning/dashboard
```

之后重启训练（或 store server），报错会被一条带 dashboard 地址的 INFO 日志取代。

## 本项目中什么时候能真正看到 dashboard

Dashboard 只在启动了 store **server** 的场景下存在：

- **Linux、Python ≤ 3.13**（`apo_train.py` 自动选择的并行 client/server 策略）：
  会为 runner 进程启动 `LightningStoreServer`。构建好 dashboard 后，用浏览器
  打开日志中打印的地址（默认端口 `4747`）即可。
- **macOS / Windows**（串行共享内存回退，见
  [README § 执行策略](../README.zh.md)）：一切都在单进程内运行，**根本没有
  HTTP 服务**，所以没有 dashboard——也不会出现 "not found" 报错，因为
  `_setup_dashboard` 从未被调用。

也可以随时单独启动一个 store server 来浏览数据：

```bash
agl store --port 4747          # 或：.venv/bin/agl store --port 4747
```

然后用浏览器访问 `http://localhost:4747/`。

### 一个服务、两行日志

训练自己启动 server 时会同时看到这两行——它们指的是**同一个** HTTP 服务
（UI 和 API 在同一端口），不是两个服务：

```
INFO  Agent-lightning dashboard will be available at http://localhost:4747
INFO  Serving the lightning store at http://localhost:4747, ...
```

### 让训练使用你自己启动的 store server

默认情况下（`managed_store=True`），client/server 策略会把 trainer 的内存
store 包装成自己的 `LightningStoreServer` 挂在 `4747` 端口。如果你已经自己
运行了 `agl store --port 4747`，自动启动的 server 会和你的端口冲突。要让训练
直接连接**你的** server：

```python
from agentlightning.store.client_server import LightningStoreClient

trainer = Trainer(
    algorithm=...,
    store=LightningStoreClient("http://localhost:4747"),
    strategy={"type": "cs", "n_runners": 4, "managed_store": False},
)
```

（也可以用环境变量 `AGL_MANAGED_STORE=false` 代替 strategy 里的参数。）
这样所有 rollout 数据都写入你的 server，dashboard 由它提供，数据还能跨
训练运行保留。

## 小结

| 场景 | 结果 |
| --- | --- |
| 出现报错，但不需要 dashboard | 直接忽略——训练不受影响 |
| 想使用 Web UI | `cd dashboard && npm install && npm run build`，然后重启 |
| 在 macOS 运行（shm 回退） | 没有 server → 没有 dashboard，构建与否都一样 |
