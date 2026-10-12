# AudioScope — L3 事件检测阶段归档

本目录是 AudioScope L3 事件检测阶段的归档。除第一版合成复音 CRNN 外，现已增加 DESED
真实强标注验证、类别独立阈值、短事件专项优化以及 CRNN–Conformer 对照。

## 当前结论

- 数据：500 个 10 秒合成复音声景、1515 个事件，最大 polyphony 为 4；
- 划分：训练/验证/测试的源录音 ID 完全隔离；
- 模型：246,762 参数 CRNN，20 ms 帧级多标签输出；
- 测试 micro event-F1：`0.248`；
- 测试 1 秒 micro segment-F1：`0.557`；
- 阈值 `0.8` 只在验证集选择。

这些分数只适用于受控合成复音协议，不代表真实录音域上的最终性能。

增强版使用 1153 条真实强标注录音，推荐 `crnn_short_event.pt`。其 overall profile 测试
event-F1/segment-F1 为 `0.258/0.593`；short_event profile 的短事件 F1 为 `0.210`。

## 目录

- `01_项目代码`：数据生成、CRNN、指标、CLI 及工程配置；
- `02_实验配置`：正式训练 YAML；
- `03_数据与时间标注`：500 条生成音频、JSONL onset/offset 标注和元数据；
- `04_实验报告`：主报告、机器可读指标、阈值扫描、逐类结果和预测；
- `05_模型文件`：最终 checkpoint、元数据和首轮消融 checkpoint；
- `06_LogMel与帧标签缓存`：三组固定特征和 frame target 缓存；
- `07_测试与文档`：单元测试、数据卡、模型卡、路线图和项目说明；
- `08_推理样例`：一条测试音频、参考/预测时间轴 CSV。
- `09_L3增强_真实强标注`：增强代码、配置、报告、模型、manifest 和校验元数据。

## 建议阅读顺序

1. `04_实验报告/ESC10_POLYPHONIC_CRNN.md`；
2. `07_测试与文档/SED_DATASET_CARD.md`；
3. `07_测试与文档/SED_MODEL_CARD.md`；
4. `02_实验配置/esc10_polyphonic_crnn.yaml`；
5. `01_项目代码/sed.py`、`sed_data.py`、`sed_metrics.py`。

## 复现入口

请在原项目根目录运行，而不是在本归档目录直接运行：

```bash
.venv/bin/python -m audioscope.cli sed-train-evaluate \
  --config configs/esc10_polyphonic_crnn.yaml
```

归档日期：2026-09-16（Asia/Shanghai）。
