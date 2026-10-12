# AudioScope 演示视频脚本 / Demo Video Script

v1.0 视频已改为自动生成：`python -m audioscope.release_video`。六段内容为分类到事件检测、
数据目录、真实推理时间轴、可控软件域偏移、有界流式测试、发布边界。使用 Tingting 中文合成
配音，内嵌中文字幕，附 `zh-CN.srt` 与 `en.srt`。实际旁白和时间由构建清单与字幕记录。
以下现场录屏脚本保留作未来 `field-final` 版本参考，不是软件版待办。

建议成片 2 分 30 秒至 3 分钟，使用同一屏幕录制一次中文或英文旁白；另一语言使用字幕。禁止把
尚未执行的现场验收剪成已完成结果。

| 时间 | 画面 | 中文旁白 | English narration |
|---|---|---|---|
| 0:00–0:15 | 标题、波形与事件时间轴 | AudioScope 把环境声音分类逐步扩展为真实多标签事件检测与 CPU 部署。 | AudioScope grows environmental sound classification into real multi-label event detection and CPU deployment. |
| 0:15–0:35 | 架构图：数据→训练→部署→目录 | 每个音频、标注、切分、模型与指标都有来源、许可、版本和哈希。 | Every audio file, annotation, split, model, and metric has traceable provenance, licensing, versioning, and hashes. |
| 0:35–1:05 | 上传长音频，运行 ONNX | 10 秒窗口以 5 秒重叠滑动，先融合帧概率再解码事件，避免窗口边界重复。 | Ten-second windows overlap by five seconds; frame probabilities are merged before decoding to avoid boundary duplicates. |
| 1:05–1:25 | 事件表、unknown 与三种阈值 | 三种阈值配置来自验证集；unknown 是置信度拒识基线，不是已解决的开放集识别。 | Three threshold profiles are selected on validation data; unknown is a confidence-rejection baseline, not solved open-set recognition. |
| 1:25–1:45 | 基准表 | 推荐 ONNX FP32：10 秒窗口中位延迟 0.873 毫秒，约 120 秒音频 RTF 0.000381。 | The recommended ONNX FP32 path reaches 0.873 ms median latency per ten-second window and 0.000381 RTF on about 120 seconds of audio. |
| 1:45–2:05 | API `/docs` 与数据库表 | PostgreSQL 保存血缘与指标，音频和模型仍以 URI 与哈希关联，服务可由 Docker Compose 重建。 | PostgreSQL stores lineage and metrics while audio and models remain referenced by URI and hash; Docker Compose rebuilds the service. |
| 2:05–2:30 | 现场采集音轨、矩阵、质检结果 | 现场只需完成九次连续录音，系统自动对齐、质检并汇总设备与房间差异。 | Field work requires nine continuous recordings; the system automatically aligns, validates, and aggregates device and room differences. |
| 2:30–2:45 | 限制页 | 当前风险包括低信噪比退化、开放集误报，以及待完成的真实设备验收。 | Current risks include low-SNR degradation, open-set false positives, and pending physical-device acceptance. |

## 录屏清单

1. `docker compose ps` 与 API 健康页；
2. `/v1/readiness` 的实际计数；
3. Gradio 上传 `l4_long_demo.wav` 后的时间轴；
4. 三种阈值切换及 unknown 说明；
5. `field_qc.json` 的真实现场结果（仅采集后录制）；
6. 结尾展示仓库结构、测试通过和限制。

字幕模板应由最终旁白自动转写后人工校时；本表中的双语句子可直接作为字幕初稿。
