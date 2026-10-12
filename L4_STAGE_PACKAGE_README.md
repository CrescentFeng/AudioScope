# AudioScope L4 工程阶段归档说明

本目录归档应包含：工程代码、配置、测试、校准部署 checkpoint、ONNX FP32/INT8、120 秒长音频
验收样例、机器可读报告、部署指南和模型卡。推荐入口是 `reports/L4_ENGINEERING.md`。

默认部署：增强 CRNN + `overall` profile + ONNX Runtime FP32。INT8 文件仅用于复现当前无收益
的量化实验；开放集输出是置信度拒识基线。
