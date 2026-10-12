# AudioScope L2：ESC-50 迁移学习实验报告

## 结论

L2 迁移学习阶段已完成。冻结预训练编码器后，只训练一个多项逻辑回归分类头，AST 在 ESC-50
官方五折达到 Accuracy `0.961 ± 0.017`、Macro-F1 `0.960 ± 0.019`；CLAP 线性探针达到
Accuracy `0.968 ± 0.007`、Macro-F1 `0.967 ± 0.008`。完全不使用 ESC-50 训练音频的 CLAP
零样本分类也达到 Accuracy `0.844 ± 0.014`、Macro-F1 `0.831 ± 0.017`。

与 L1 clean LogMel CNN 相比，AST 与 CLAP 线性探针的 Macro-F1 分别提高约 48.0 和 48.7 个
百分点。这个增益来自大规模预训练表征，不等价于“仅靠 ESC-50 学到”的能力。

## 实验协议

- 数据：ESC-50，2,000 条、50 类、每类 40 条、每条约 5 秒。
- 划分：官方 5 folds；每次用 4 folds 拟合分类头，剩余 1 fold 测试。
- AST：`MIT/ast-finetuned-audioset-10-10-0.4593`，冻结 768 维编码器。
- CLAP：`laion/clap-htsat-unfused`，冻结 512 维音频编码器。
- 线性探针：训练折 StandardScaler + multinomial logistic regression，`C=1.0`。
- 零样本：每类三个预先固定的英文模板，文本嵌入平均后与音频嵌入做余弦相似度。
- 样本效率：AST 每折训练集按类别和固定种子哈希排序，取每类 1/2/4/8/16/32 条。
- 指标：Accuracy、Macro-F1；报告五折总体均值与总体标准差。
- 泄漏说明：标准化器和分类头只拟合训练折；但官方 ESC-50 本身存在 4 个跨 fold 的来源 ID，
  为和公开协议一致而保留，沿用 L1 已披露的限制。

## 主结果

| 方法 | ESC-50 监督量 | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| MFCC + SVM（L1） | 完整训练折 | 0.373 ± 0.031 | 0.312 ± 0.027 |
| LogMel CNN（L1） | 完整训练折 | 0.505 ± 0.041 | 0.480 ± 0.041 |
| CLAP 零样本 | 0 条 | 0.844 ± 0.014 | 0.831 ± 0.017 |
| AST 线性探针 | 完整训练折 | 0.961 ± 0.017 | 0.960 ± 0.019 |
| CLAP 线性探针 | 完整训练折 | 0.968 ± 0.007 | 0.967 ± 0.008 |

AST 各折 Macro-F1：`0.949 / 0.990 / 0.955 / 0.970 / 0.935`。

CLAP 零样本各折 Macro-F1：`0.858 / 0.804 / 0.831 / 0.832 / 0.831`；CLAP 线性探针：
`0.954 / 0.975 / 0.965 / 0.975 / 0.966`。

## AST 样本效率

| 每类训练样本 | 每折训练总数 | Accuracy | Macro-F1 |
|---:|---:|---:|---:|
| 1 | 50 | 0.820 ± 0.033 | 0.804 ± 0.038 |
| 2 | 100 | 0.886 ± 0.020 | 0.880 ± 0.021 |
| 4 | 200 | 0.923 ± 0.015 | 0.919 ± 0.016 |
| 8 | 400 | 0.945 ± 0.017 | 0.943 ± 0.017 |
| 16 | 800 | 0.956 ± 0.018 | 0.955 ± 0.019 |
| 32 | 1,600 | 0.961 ± 0.017 | 0.960 ± 0.019 |

每类 8 条已经达到完整训练折表现的约 98.3%（按 Macro-F1 比值），表明预训练表示显著降低了
ESC-50 的标签需求。

## 计算成本

运行环境为 Apple arm64、macOS 26.6.2、Python 3.12.14、PyTorch 2.14.0、Transformers
5.17.0、scikit-learn 1.9.1。

| 阶段 | 设备 | 批量 | 墙钟时间 |
|---|---|---:|---:|
| AST 提取 2,000 条嵌入 | MPS | 4 | 218.6 s |
| AST 五折完整线性探针训练 | CPU | - | 0.27 s |
| CLAP 提取 2,000 条嵌入 | CPU | 4 | 79.9 s |
| CLAP 五折线性探针训练 | CPU | - | 0.23 s |

提取完成后，AST 768 维缓存约 5.9 MB，CLAP 512 维缓存约 4.0 MB；后续重跑分类头无需再次做
音频编码。时间只代表本机单次实测，不应外推为其他机器的延迟基准。

## 错误分析

AST 线性探针最低召回类别是 `helicopter`（0.65），主要被判为 `washing_machine`（8 条）或
`airplane`（5 条）；其次是 `footsteps`（0.85），常与 `door_wood_knock` 混淆。

CLAP 零样本的主要弱点是细粒度语义和相似声学纹理：`wind` 召回 0.15，`hen` 0.20，
`crickets` 0.275；最多的错误对是 `crickets → frog`（25 条）、`airplane → helicopter`
（21 条）和 `hen → rooster`（16 条）。加入 ESC-50 线性探针后，CLAP 的最低召回提升为
`helicopter` 0.80，说明少量有监督映射能有效校正通用文本语义与本数据集标签边界之间的偏差。

## 可复现性与边界

- AST revision：`f826b80d28226b62986cc218e5cec390b1096902`；权重 SHA-256：
  `ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d`。
- CLAP revision：`8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a`；权重 SHA-256：
  `1cd3c601bc4afe0fa87be3de4c13dd2cfadd249fac1e29acf74a9b296c3219bb`。
- AST 为 BSD-3-Clause，CLAP 为 Apache-2.0；ESC-50 为 CC BY-NC 3.0，因此本阶段结果按研究与
  作品集用途管理。
- AudioSet 与 ESC-50 类别语义重叠，CLAP 训练语料也可能覆盖相近概念；高分代表迁移能力，
  不能被表述为从零学习或严格域外泛化。
- 本阶段仍是闭集、单标签、片段级分类；未知声音拒识、事件时间定位、真实设备噪声与流式部署
  属于后续阶段。

## 复现命令

```bash
.venv/bin/python -m pip install -e '.[transfer,dev]'
.venv/bin/python -m pytest
.venv/bin/python -m audioscope.cli transfer-evaluate \
  --config configs/esc50_ast_linear_probe.yaml
.venv/bin/python -m audioscope.cli transfer-evaluate \
  --config configs/esc50_clap.yaml
```

机器可读主结果位于 `reports/generated/esc50-ast-linear-probe/transfer_summary.json` 与
`reports/generated/esc50-clap/transfer_summary.json`；每折目录包含指标、混淆矩阵和错误样例。
