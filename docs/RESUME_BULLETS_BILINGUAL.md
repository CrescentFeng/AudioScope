# 简历要点 / Resume Bullets

v1.0 新增可引用结果：在 221 条测试录音上完成七种确定性软件域偏移及重复推理哈希校验；
将流式事件解码改为固定缓存状态，单处理器通过 1,800 秒音频加速压力测试，并自动生成中文配音
双语字幕演示。未进行自有物理设备/房间认证。

v1.0: evaluated seven deterministic software domain shifts on 221 test recordings with repeat-inference
hash checks; implemented bounded incremental event decoding and passed an accelerated 1,800-second
audio stream test; generated a narrated demonstration with bilingual subtitles. Physical device and room
certification has not been performed.

## 中文

- 构建环境音频端到端系统，覆盖 ESC-50 分类、DESED 真实强标注事件检测、长音频/流式推理、
  开放集拒识与 ONNX CPU 部署；以源录音隔离切分避免派生片段泄漏。
- 在 Apple CPU 上将推荐 ONNX FP32 路径做到 10 秒窗口 P50/P95 约 0.873/0.967 ms、120 秒音频
  端到端 RTF 0.000381，并如实记录动态 INT8 未量化 GRU/卷积且无加速的负向实验。
- 设计 PostgreSQL 数据血缘模型与 Alembic 迁移，幂等导入 3,153 条音频元数据、6,224+ 条标注、
  模型哈希与基准指标；通过 FastAPI 和 Docker Compose 提供可复现查询与部署。
- 为 3 设备×3 房间实验生成带同步标记的连续播放协议、自动对齐/质检/指标汇总与最终发布门禁，
  将现场人工采集与后续自动处理清晰分离。

## English

- Built an end-to-end environmental audio system spanning ESC-50 classification, real strongly labelled
  DESED event detection, long-form and streaming inference, open-set rejection, and ONNX CPU deployment,
  with source-disjoint splits preventing derived-clip leakage.
- Achieved approximately 0.873/0.967 ms P50/P95 latency per ten-second window and a 0.000381 end-to-end
  RTF on a 120-second file with ONNX FP32 on an Apple CPU; documented that dynamic INT8 did not quantize
  the GRU/convolutions and delivered no speedup.
- Designed a PostgreSQL lineage schema and Alembic migration with idempotent ingestion for 3,153 audio
  records, 6,224+ annotations, model hashes, and benchmark metrics, exposed through FastAPI and Docker
  Compose.
- Developed a synchronized 3-device × 3-room capture protocol with automatic alignment, quality checks,
  metric aggregation, hardware acceptance logging, and an explicit final-release gate.

## 工作台增强可引用要点 / Workbench enhancement bullets

- 实现可回听事件表、预测/参考双时间轴、追加式复核版本、冲突保护与结果导出；通过真实 PostgreSQL、
  ONNX、CLAP 容器链路验收，人工/软件复核与训练数据严格分开。
- 仅用验证集选择事件解码参数，在既有 221 条 DESED test 上将 clean event-F1 从 0.2583 提高至
  0.2873，六种软件扰动平均值从 0.2210 提高至 0.2568，同时披露召回率下降和三种子增强负向实验。
- Implemented replayable event review, aligned prediction/reference timelines, append-only corrections,
  optimistic conflict checks and exports, verified against real PostgreSQL/ONNX/CLAP container services.
- Improved clean event-F1 from 0.2583 to 0.2873 and mean six-shift event-F1 from 0.2210 to 0.2568 on an
  existing 221-clip test split using validation-only decoder selection; disclosed reduced recall and
  rejected augmentation weights rather than presenting only positive experiments.

上述指标限于当前数据协议和软件条件，不代表真实设备、现场环境或通用声音识别认证。
These results are protocol-specific software evidence, not physical-device or general-purpose certification.
