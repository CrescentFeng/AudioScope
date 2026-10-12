# AudioScope L4 工程阶段：长音频、流式、开放集与 ONNX

## 结论

L4 已把 L3 增强 CRNN 做成可复现的 CPU 部署链路：支持任意时长 WAV 的 10 秒滑窗与 5 秒
重叠融合、1 秒块流式推理、麦克风输入、验证集温度校准、未知类别置信度拒识、ONNX 导出、
动态量化对照、全量测试集 SNR 曲线，以及上传/录音 Gradio 界面。

推荐部署组合是 `crnn_short_event_calibrated.pt` + ONNX FP32。Apple CPU 实测中，ONNX FP32
处理一个 10 秒窗口的中位模型延迟约 `0.87 ms`，120 秒长音频端到端 RTF 约
`0.00038`。当前 INT8 路径没有量化到 GRU 和卷积，既未显著减小体积，也未提速，因此不作为
推荐部署格式。

| 后端 | 10 秒窗口 P50 / P95 | 120 秒端到端 RTF | 模型文件 | 权重字节 |
|---|---:|---:|---:|---:|
| PyTorch FP32 | 3.284 / 3.773 ms | 0.000815 | 1,015,338 B | 987,048 B |
| **ONNX Runtime FP32** | **0.873 / 0.967 ms** | **0.000381** | 992,343 B | 986,152 B |
| ONNX Runtime dynamic INT8 | 0.888 / 0.921 ms | 0.000388 | 991,893 B | 982,317 B |

## 1. 部署模型与校准

部署源模型是 L3 的 `crnn_short_event.pt`。L4 不覆盖源 checkpoint，而是生成独立部署包。
在 229 条 DESED validation 录音上拟合单一温度：

- 温度：`0.92653`；
- multi-label NLL：`0.29876 → 0.29797`；
- 10-bin ECE：`0.15704 → 0.15122`；
- Brier：`0.09297 → 0.09307`，略微变差；
- 校准后重新只在 validation 上选择三组事件阈值，overall 仍为 `0.8`。

温度缩放改善有限，不能把本模型解释为已经获得严格概率校准。当前部署保留完整的校准前后
数据，而不是只报告改善项。

## 2. 长音频与跨窗融合

长音频按 10 秒窗口、5 秒 hop 推理。每个窗口输出 20 ms 帧级概率；重叠区使用带 0.1 边界
下限的三角权重融合，最后只做一次事件解码。这比先解码每个窗口再拼事件更稳定，也避免同一
事件在窗口接缝被重复输出。

验收音频由 12 条 test 录音按确定顺序拼接，长度 `119.361 s`，不参与训练或阈值选择：

- 23 个滑窗；
- micro event-F1 `0.232`；
- 1 秒 micro segment-F1 `0.638`；
- ONNX FP32 端到端耗时约 `0.0455 s`，RTF `0.000381`。

该拼接测试用于验证时间偏移、尾段覆盖和跨窗融合；它不是新的独立泛化数据集。

## 3. 流式与麦克风路径

流式处理器每 1 秒运行一次，只保存最近 10 秒音频历史；事件在 0.4 秒稳定等待后输出，结束时
flush 未完成事件。120 秒流式模拟的 ONNX 计算延迟中位数 `2.116 ms`、P95 `2.423 ms`。
算法延迟由 1 秒采集块主导，而不是模型前向。最近一次构建的计算 P95 为 `2.16 ms`。CLI 提供
真实 `sounddevice` 麦克风采集；自动测试
只验证文件流模拟，因为无头环境不能替代实际声卡验收。

## 4. 开放集基线

开放集使用“最大已知类概率”拒识。正样本来自 DESED validation 的标注活动帧，域外样本来自
ESC-50 中不属于部署 10 类的 200 条录音，低于 -50 dBFS 的帧不参与拟合。

| 指标 | 结果 |
|---|---:|
| 已知置信度阈值 | 0.825 |
| AUROC | 0.750 |
| 平衡准确率 | 0.674 |
| 已知召回 | 0.795 |
| 未知召回 | 0.553 |
| 95% 已知召回下 FPR | 0.785 |

这只是 confidence rejection baseline，不是学习到的 unknown 类。FPR@95 很高，说明模型对域外
声音仍会过度自信；简历中应写“实现并评估开放集基线”，不能写“解决未知类别”。

## 5. ONNX 与量化

ONNX 使用 opset 17，输入 `[batch, 1, 64, 500]`，输出 `[batch, 500, 10]`。当前 PyTorch
`dynamo=True` 导出器会把 GRU 的 batch 特化为 1，因此构建流程检测到该问题后回退到 legacy
导出器。最终模型已实际通过 batch=2 运行验证。

- PyTorch–ONNX 最大 logit 绝对误差：`2.38e-6`；
- `atol=rtol=1e-4` 下完全一致；
- ONNX checker 通过。

动态 INT8 请求量化 MatMul/Gemm/GRU，但当前 ONNX Runtime 图中只出现 1 个
`MatMulInteger`，GRU 和 Conv 仍为 FP32。文件仅缩小 `0.045%`，最大 logit 误差
`0.0215`，且窗口延迟略慢。因此保留产物用于复现负向结果，不推荐上线。

## 6. 不同 SNR 的真实测试集结果

对全部 221 条 DESED test 录音在波形层加入确定性白噪声，每档总音频 `2183.5 s`：

| 条件 | micro event-F1 | 1 秒 segment-F1 |
|---|---:|---:|
| clean | 0.258 | 0.585 |
| 20 dB | 0.215 | 0.510 |
| 10 dB | 0.155 | 0.390 |
| 0 dB | 0.031 | 0.192 |
| -5 dB | 0.023 | 0.128 |

结果显示噪声泛化是当前最明显的模型风险。L4 没有用测试噪声反向调参；后续改进应在训练集引入
真实噪声和混响，并另设验证条件。

## 7. 复现

```bash
.venv/bin/python -m pip install -e '.[engineering,demo,microphone,dev]'
.venv/bin/audioscope l4-build --config configs/l4_engineering.yaml

.venv/bin/audioscope sed-long-predict \
  --model artifacts/l4-engineering/crnn_short_event_calibrated.pt \
  --onnx artifacts/l4-engineering/crnn_short_event_fp32.onnx \
  --audio data/l4-engineering/l4_long_demo.wav

.venv/bin/audioscope sed-stream-file \
  --model artifacts/l4-engineering/crnn_short_event_calibrated.pt \
  --onnx artifacts/l4-engineering/crnn_short_event_fp32.onnx \
  --audio data/l4-engineering/l4_long_demo.wav

.venv/bin/python l4_app.py \
  --model artifacts/l4-engineering/crnn_short_event_calibrated.pt \
  --onnx artifacts/l4-engineering/crnn_short_event_fp32.onnx
```

机器可读结果位于 `reports/generated/l4-engineering/`。部署说明见
`docs/L4_DEPLOYMENT.md`，责任边界见 `docs/L4_MODEL_CARD.md`。

## 8. 未完成与限制

1. 真实麦克风 CLI 已实现，但最终声卡、系统权限和现场声学条件需要人工验收；
2. 还没有真实噪声、混响和设备迁移曲线；
3. 开放集只是置信度拒识，AUROC 与 FPR 尚不足以称为可靠 OOD 检测；
4. INT8 在当前 CRNN/ONNX Runtime 组合上无收益；
5. RSS 增量受同进程分配器缓存影响，报告以模型文件与权重字节为稳定内存代理；
6. 数据库、Docker、现场采集自动化、双语摘要和视频脚本已补齐；真实设备/房间录音、实体麦克风
   与最终视频录制仍必须由操作者在现场完成，发布审计不会把模拟结果当作现场证据。
