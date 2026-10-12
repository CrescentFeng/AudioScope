# AudioScope - L1 基础阶段 / LogMel CNN 交付包

归档日期：2026-09-15

本交付包代表 L1 的 LogMel CNN 部分。它与 `01_传统基线` 一起构成完整的 L1 基础阶段，
不代表 AudioScope 的 L2-L4 已完成。

## 已完成内容

- 固定尺寸 log-mel 特征与可复用缓存
- 36,370 参数的小型 2D CNN
- 固定随机种子、AdamW、label smoothing 与 SpecAugment
- ESC-50 官方五折 clean CNN 评测
- 20/10 dB 波形噪声训练消融及 20/10/0 dB 鲁棒性评测
- 每折模型、训练曲线、混淆矩阵、错误样例和 Top-3 推理
- CNN smoke 测试与 CI

Clean CNN：Accuracy `0.505 ± 0.041`，Macro-F1 `0.480 ± 0.041`。

Noise-augmented CNN：clean Macro-F1 `0.468 ± 0.033`，20/10/0 dB Macro-F1 分别为
`0.416 / 0.323 / 0.101`。

## 文件分类

- `01_项目代码`：CNN、log-mel 与 CLI 源码
- `02_实验配置`：smoke、clean 五折与噪声增强五折配置
- `03_实验报告`：阶段报告及每折原始结果
- `04_模型文件`：clean、噪声增强和 smoke 模型
- `05_LogMel缓存`：干净与噪声 log-mel 复现缓存
- `06_测试与CI`：CNN 单元/端到端测试和 CI
- `07_项目文档`：CNN 模型卡与路线图

ESC-50 原始数据和 `.venv` 不重复归档；位置与版本沿用传统基线交付包中的说明。

