# AudioScope L3 增强阶段归档

本目录对应 L3 的真实强标注增强实验，包含 DESED 强标注划分、类别独立阈值、短事件损失与
工作点、CRNN–Lite Conformer 对照，以及完整模型和机器可读报告。

## 推荐结果

- 模型：`05_模型文件/crnn_short_event.pt`；
- overall profile：test micro event-F1 `0.258`，segment-F1 `0.593`；
- short_event profile：test 短事件 F1 `0.210`，整体 event-F1 `0.274`；
- 默认 CLI profile 为 `overall`，短事件任务显式选择 `short_event`；
- `class_balanced` profile 保留用于分析，因 test 表现不稳定，不是默认选择。

## 目录

- `01_项目代码`：DESED 导入、CRNN/Lite Conformer、短事件损失、阈值与指标；
- `02_实验配置`：确定性实验 YAML；
- `03_真实强标注数据`：1153 条强标注录音、manifest 和 40 分片 MD5 元数据；
- `04_LogMel与帧标签缓存`：703/229/221 三组固定缓存；
- `05_模型文件`：三个模型及 threshold profiles；
- `06_实验报告`：主报告、逐模型阈值扫描、逐类指标、预测与比较 CSV；
- `07_测试与文档`：数据卡、模型卡、路线图、来源登记和测试；
- `08_推理样例`：真实录音和 reference/prediction 时间轴。

## 原始下载分片

40 个 Zenodo tar 源分片共约 847 MB，保留在原项目的
`data/raw/desed-webdataset/shards/`，没有在桌面归档中重复复制。每个分片的实际 MD5 已写入
`03_真实强标注数据/desed-real-strong-only/metadata.json`，并与 Zenodo 记录全部匹配。

## 复现

```bash
.venv/bin/python -m audioscope.cli sed-enhance-evaluate \
  --config configs/desed_real_enhanced.yaml
```

归档日期：2026-09-16（Asia/Shanghai）。
