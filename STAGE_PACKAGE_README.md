# AudioScope - L1 基础阶段 / 传统基线交付包

归档日期：2026-09-15

## 当前完成状态

这个交付包只代表 L1 的传统基线部分，不代表完整 AudioScope，也不代表 L1 中尚未完成的
log-mel + CNN 部分。

已完成内容包括：

- ESC-50 官方五折数据 manifest、SHA-256 校验、来源和许可记录
- 数据 EDA、MFCC 特征、标准化 RBF SVM 和 sigmoid 概率校准
- Accuracy、Macro-F1、混淆矩阵、错误样例和白噪声鲁棒性评测
- 官方 fold 中跨 fold 来源重叠检查与 source-disjoint 敏感性分析
- 合成 smoke 数据、端到端测试、命令行入口和 CI

正式五折结果：Accuracy `0.373 ± 0.031`，Macro-F1 `0.312 ± 0.027`。

## 文件分类

- `01_项目代码`：Python 包、CLI、基础 demo、依赖与构建入口
- `02_实验配置`：smoke、官方五折及 source-disjoint 配置
- `03_数据清单与许可`：ESC-50 manifest、校验值、来源与许可台账
- `04_实验报告`：正式报告、每折指标、混淆矩阵和错误样例
- `05_模型文件`：五折模型、source-disjoint 对照模型和 smoke 模型
- `06_复现缓存`：已生成的 MFCC 特征缓存
- `07_Smoke验证数据`：只用于验证工程链路的合成 WAV
- `08_测试与CI`：单元测试、端到端测试和 GitHub Actions 配置
- `09_项目文档`：模型卡与后续路线图

## 未重复归档的外部内容

ESC-50 原始仓库约 1.4 GB，不属于阶段生成文件，因此没有在桌面重复复制。工作副本位于：

`<project-root>/data/raw/ESC-50-master`

固定数据版本：`33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6`，许可为 CC BY-NC 3.0。

项目虚拟环境 `.venv` 也没有复制，因为它可以由 `pyproject.toml` 重新创建，并且不属于项目成果。

## 原始可运行工作区

桌面目录是按阶段整理的交付副本。后续开发仍以以下目录为准：

`<project-root>`
