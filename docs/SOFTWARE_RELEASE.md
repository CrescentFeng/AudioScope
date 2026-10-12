# AudioScope v1.0 Software-Validated Release

软件验证完成；未进行自有物理设备/房间认证。
Physical device and room certification has not been performed.

## 版本范围与实测结果

采用全部 221 条 DESED test，固定种子 2027 和原验证集确定的 overall 阈值。每种条件重复
变换及推理，输出事件哈希完全一致。变换不修改源数据，模拟混响裁至原长度，保持原始真值
时间；峰值超过 0.98 时统一缩放。这里的 RT60 是模拟衰减参数，不是实测房间参数。

| 条件 | micro event-F1 |
|---|---:|
| clean | 0.2583 |
| 模拟 RT60 0.25s | 0.2531 |
| 模拟 RT60 0.8s | 0.2436 |
| 电话带宽 300–3400 Hz | 0.1953 |
| 模拟笔记本带宽与非线性 | 0.2303 |
| 10 dB 粉红噪声 | 0.2165 |
| 长混响＋设备响应＋粉红噪声 | 0.1871 |

完整 macro event-F1、segment-F1、相对 clean 退化、时间及 RTF 见机器可读 `robustness.json`。
评测通过表示覆盖完整、数值有效且重复一致，不意味着模型在所有扰动下均有高准确率。

单个流式处理器以一秒块连续处理 1,800 秒音频，耗时 10.856 秒，RTF 0.00603，块计算 P95
9.62 ms，预热后 RSS 增长 39.03 MiB。内部最多保留 160,000 个音频样本、3 帧概率平滑窗口、
10 个类别状态和 2,048 个延迟值。长事件结束才输出；跨类别按可确认顺序输出，每类时间单调。
10,000 块回归测试验证状态上界。报告测试的是加速文件流，不是 30 分钟墙钟时间声卡稳定性。

首轮同时运行全量域偏移评测时，压力测试进程 RSS 增长约 192 MiB，超出预算；独立复测增长约
39 MiB，通过 64 MiB 标准。负载及运行时分配会影响测量，因此发布基准使用独立运行结果，
不宣称并行高负载条件也满足同一 RSS 预算。

## 复现

```bash
.venv/bin/python -m pip install -e '.[engineering,release,database,api,dev]'
.venv/bin/python scripts/verify_software_release.py
.venv/bin/python -m audioscope.software_validation evaluate
.venv/bin/python -m audioscope.software_validation soak --seconds 1800
.venv/bin/python -m audioscope.release_catalog snapshot
.venv/bin/python -m audioscope.release_video
.venv/bin/python -m audioscope.release_catalog sync
.venv/bin/python -m audioscope.release_audit --mode software-final
.venv/bin/python -m audioscope.release_catalog sync
.venv/bin/python -m audioscope.release_audit --mode field-final
```

最后一条预期因缺少现场证据而失败。`final` 保持 `field-final` 语义，`preflight` 继续保留。
第二次 catalog sync 将已生成的软件发布审计同步进数据库；重复运行按证据内容哈希幂等写入。
正式视频依赖 macOS `say` 的 Tingting、Pillow、FFmpeg 和 ffprobe。CI 生成短片使用测试音调，
明确标记 smoke，不能用于正式发布。生成视频不包含任何数据集原始音频，只展示指标和事件图。

## 交付与限制

报告在 `reports/generated/software-release/`；视频、双语 SRT 和构建清单在 `release/video/`；
软件与现场审计分别保存在 `reports/generated/release/`。PostgreSQL 保存报告路径、内容哈希、
处理运行、指标、视频清单和发布审计，媒体内容不写入 SQL。

数据库及兼容 API 容器已经验证；正式 Python/FastAPI Docker 镜像的在线构建仍受先前镜像仓库
连通性限制，不能以兼容 PHP API 的通过结果替代正式镜像验收。模型仍存在低信噪比退化、
开放集误报和类别覆盖限制。实体录音协议保留为未来 `field-final` 扩展。
