# AudioScope 项目摘要 / Project Summary

发布名称：AudioScope v1.0 Software-Validated Release。
软件验证完成；未进行自有物理设备/房间认证。
Physical device and room certification has not been performed.

软件验证采用 221 条测试录音的七种可控扰动、固定种子与重复推理校验；有界流式状态通过
1,800 秒加速音频压力测试。演示视频由实际报告、事件时间轴与中文合成语音自动生成。
These software experiments use seven deterministic perturbations on 221 test clips, repeat-inference
checks, and an accelerated 1,800-second stream test with bounded state. The video uses measured reports,
stored event predictions and synthetic Chinese narration. Physical validation remains optional future work.

## 中文

AudioScope 是一个从环境声音分类扩展到多标签声音事件检测与 CPU 部署的端到端项目。项目在
ESC-50 上建立 MFCC+SVM、LogMel CNN、AST 与 CLAP 基线；在 1,153 条 DESED 真实强标注录音
上以源录音隔离方式训练并比较 CRNN 与轻量 Conformer；随后补齐类别独立阈值、短事件损失、
置信度校准、开放集拒识、长音频重叠融合、1 秒块流式推理和 ONNX Runtime 部署。

当前推荐的 ONNX FP32 模型在本机 CPU 上处理 10 秒窗口的中位模型延迟为 0.873 ms，约 120 秒
长音频的端到端 RTF 为 0.000381；开放集置信度基线 AUROC 为 0.750。系统同时提供 PostgreSQL
数据血缘目录、Alembic 迁移、幂等数据导入、只读 FastAPI、Docker Compose、真实设备/房间
采集协议、同步与自动质检、硬件验收记录器和发布门禁。真实设备/房间结果须在现场采集后填入，
未采集前不把文件流模拟表述为硬件结论。

## English

AudioScope is an end-to-end environmental audio project that progresses from clip-level classification to
multi-label sound event detection and CPU deployment. It establishes MFCC+SVM, LogMel CNN, AST, and
CLAP baselines on ESC-50, then trains and compares a CRNN and a lightweight Conformer on 1,153 real,
strongly labelled DESED recordings using source-disjoint splits. The deployment layer adds class-specific
thresholds, short-event optimization, confidence calibration, open-set rejection, overlap-merged long-audio
inference, one-second streaming, and ONNX Runtime export.

The recommended ONNX FP32 model reaches 0.873 ms median model latency for a ten-second window and an
end-to-end real-time factor of 0.000381 on an approximately 120-second file on the measured CPU. The
confidence-rejection open-set baseline reaches 0.750 AUROC. The project also includes a PostgreSQL lineage
catalog, Alembic migrations, idempotent ingestion, a read-only FastAPI service, Docker Compose, a controlled
device/room capture protocol, synchronization and automated quality checks, a hardware acceptance recorder,
and a release gate. Device and room results remain explicitly pending until physical capture is performed;
file simulation is never presented as hardware evidence.

## 后续工作台增强 / Subsequent workbench enhancement

新增可回听、可编辑且有版本保护的事件工作台，原始预测与复核结果分开保存；提供预测/参考双时间轴、
同音频模型对照、历史与 JSON/CSV 导出。验证集选择的解码配置将现有 221 条 test 的 clean event-F1
从 0.2583 提高至 0.2873，但召回率由 0.3925 降至 0.3642；六种可控扰动的平均 event-F1
从 0.2210 提高至 0.2568。三种子增强训练不满足 clean 容忍门槛，未部署其权重。

本地 CLAP 提供英文描述候选检索，固定十类查询中有阳性的 97 次查询平均 P@5 为 0.6598；
目标不存在时仍可能返回候选，不宣称通用语义识别。正式 Python Docker、迁移、真实 ONNX/CLAP
工作流已通过本机验收；整体新版发布门禁与归档状态以 `docs/UPGRADE_STATUS.md` 为准。
仍未进行自有物理设备/房间认证。

The enhancement adds replayable events, versioned corrections separate from immutable predictions,
aligned reference timelines, same-audio model comparisons, history, and JSON/CSV exports. A
validation-selected decoder improves clean event-F1 from 0.2583 to 0.2873 on the existing 221-clip test
split, with lower recall (0.3925 to 0.3642), and improves mean event-F1 across six controlled shifts
from 0.2210 to 0.2568. Three-seed augmentation weights failed the clean-validation tolerance and were
not deployed. Local English CLAP retrieval achieves mean P@5 of 0.6598 on 97 positive queries in a
fixed ten-class benchmark; absent sounds can still return candidates. Python Docker, migrations and
real ONNX/CLAP workflows have passed local acceptance. Overall enhancement release/archival status
is tracked separately. Physical device and room certification has not been performed.
