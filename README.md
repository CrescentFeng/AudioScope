# AudioScope

AudioScope 是一个可复现的环境声音理解与声音事件检测项目，覆盖传统分类基线、迁移学习、
CRNN/Conformer 事件检测、长音频流式推理、开放集评测、数据库与 Docker 工作台。

**当前版本：v1.0.0 软件验证版。** 事件回听与复核、预测/参考双时间轴、模型对照、历史导出及
本地文字检索均已实现。容器工作台默认入口为 `http://127.0.0.1:7862`，API 文档为
`http://127.0.0.1:8002/docs`。启动命令、音频限制和隐私说明见
[工作台使用说明](docs/WORKBENCH_GUIDE.md)，
实验及负面结果见 [增强报告](reports/WORKBENCH_IMPROVEMENT.md)，
增强发布证据见 [发布记录](docs/WORKBENCH_RELEASE.md)。包版本保留 1.0.0，增强版以证据哈希区分。

```bash
# 在已经安装依赖并拥有本地模型的项目目录中：
.venv/bin/python -m audioscope.launch
# 或启动正式 Python 容器（先 docker compose build api）：
AUDIOSCOPE_API_PORT=8002 AUDIOSCOPE_DEMO_PORT=7862 docker compose --profile demo up -d --no-build api demo
```

边界优化使现有 221 条测试录音 clean event-F1 从 0.2583 提高到 0.2873，但召回率下降；
六种软件扰动平均 event-F1 从 0.2210 提高到 0.2568。不是通用正确率或现场认证。
人工复核不自动进入训练。CLAP 仅为英文候选检索，目标声音不存在时仍可能返回结果。

原封板发布：**AudioScope v1.0 Software-Validated Release**。软件验证完成；未进行自有物理设备/房间
认证。新增七种可控域偏移、1,800 秒有界流式压力测试、中文配音双语字幕视频及独立软件发布门禁。
完整结果和复现命令见 `docs/SOFTWARE_RELEASE.md`；实体现场验证保留为未来增强。

项目最初建立了离线 smoke 链路。它能生成确定性的合成音频，验证许可与
manifest、检查数据泄漏、完成 EDA、训练 MFCC + SVM，并导出 Accuracy、Macro-F1、混淆矩阵、
错误样例和不同 SNR 下的鲁棒性结果。官方 ESC-50 五折传统基线也已完成：Accuracy
`0.373 ± 0.031`，Macro-F1 `0.312 ± 0.027`。LogMel CNN 五折基线也已完成：Accuracy
`0.505 ± 0.041`，Macro-F1 `0.480 ± 0.041`。L2 迁移学习也已完成：AST 线性探针
Macro-F1 `0.960 ± 0.019`，CLAP 零样本 `0.831 ± 0.017`，CLAP 线性探针
`0.967 ± 0.008`。L3 已建立重叠事件 CRNN 基线：micro event-F1 `0.248`，1 秒
micro segment-F1 `0.557`。L3 增强版进一步使用 DESED 真实强标注池：增强 CRNN 的 overall
event-F1 `0.258`、segment-F1 `0.593`；short_event profile 的短事件 F1 为 `0.210`。
L4 已完成长音频、流式、开放集与 ONNX 工程闭环：ONNX FP32 的 10 秒窗口中位延迟约
`0.87 ms`，120 秒端到端 RTF 约 `0.00038`；开放集 baseline AUROC `0.750`。当前 INT8
没有量化到 GRU/卷积，不作为推荐格式。
合成数据结果只证明对应受控协议，真实数据结果也受当前确定性 split 和类别不均衡限制。

## 60 秒架构

```text
WAV + source/license metadata
            |
            v
 JSONL manifest ----> checksum + source-fold leakage checks
            |
            +-------> EDA: class / fold / duration / dBFS / anomalies
            |
            v
 resample + mono -> MFCC mean/std -> StandardScaler -> SVM
                                               |
                                               v
                         metrics + confusion matrix + errors + SNR curve
```

## 快速开始

需要 Python 3.10+。GitHub 仓库不包含第三方原始音频、本机数据库、密码、预训练权重和实验
checkpoint；数据与模型获取规则见 [数据政策](data/README.md) 和
[GitHub 发布说明](docs/GITHUB_RELEASE.md)。首次安装与基础验证：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest
.venv/bin/python -m audioscope.cli smoke --config configs/smoke.yaml
```

输出位于 `artifacts/smoke/` 和 `reports/generated/smoke/`，两者均被 Git 忽略。可用任意生成的
WAV 检查推理：

```bash
.venv/bin/python -m audioscope.cli predict \
  --model artifacts/smoke/model.joblib \
  --audio data/smoke/low_tone/low_tone-00.wav
```

安装 demo 依赖后可启动上传/录音界面：

```bash
.venv/bin/python -m pip install -e '.[demo]'
.venv/bin/python app.py --model artifacts/smoke/model.joblib
```

## 接入 ESC-50

1. 先阅读 `data/sources.yaml` 中的许可记录。ESC-50 是 CC BY-NC 3.0。
2. 将官方仓库放到 `data/raw/ESC-50-master/`（音频不会进入 Git）。
3. 生成并校验 manifest：

```bash
.venv/bin/python -m audioscope.cli prepare-esc50 \
  --dataset-root data/raw/ESC-50-master \
  --output data/manifests/esc50.jsonl
.venv/bin/python -m audioscope.cli validate-manifest \
  --config configs/esc50_mfcc_svm.yaml
```

4. 一条命令运行五个官方 fold；特征缓存会避免重复提取干净 MFCC：

```bash
.venv/bin/python -m audioscope.cli cross-validate \
  --config configs/esc50_mfcc_svm.yaml
```

完整协议、逐 fold 结果、噪声鲁棒性和错误分析见 `reports/ESC50_MFCC_SVM_BASELINE.md`。

## LogMel CNN

安装 PyTorch 并运行 smoke 或正式五折：

```bash
.venv/bin/python -m pip install -e '.[cnn]'
.venv/bin/python -m audioscope.cli cnn-cross-validate --config configs/cnn_smoke.yaml
.venv/bin/python -m audioscope.cli cnn-cross-validate --config configs/esc50_logmel_cnn.yaml
.venv/bin/python -m audioscope.cli cnn-cross-validate \
  --config configs/esc50_logmel_cnn_noise_augmented.yaml
```

CNN 完整报告见 `reports/ESC50_LOGMEL_CNN_BASELINE.md`。Clean CNN 是干净条件主结果；噪声增强
CNN 牺牲约 1.2 个 Macro-F1 百分点，将 20/10 dB Macro-F1 提升到 `0.416 / 0.323`。

## L2 迁移学习

安装迁移学习依赖后运行冻结编码器实验：

```bash
.venv/bin/python -m pip install -e '.[transfer]'
.venv/bin/python -m audioscope.cli transfer-evaluate \
  --config configs/esc50_ast_linear_probe.yaml
.venv/bin/python -m audioscope.cli transfer-evaluate \
  --config configs/esc50_clap.yaml
```

AST 路线包含官方五折线性探针和每类 1/2/4/8/16/32 条样本效率曲线；CLAP 路线包含固定
提示集成的零样本分类及五折线性探针。完整协议、成本、错误分析与边界见
`reports/ESC50_TRANSFER_LEARNING.md`。预训练权重不提交 Git，固定 revision 与 SHA-256 记录在
`data/sources.yaml`。

## L3 多标签事件检测

L3 将 ESC-10 不同官方 folds 的源录音组合为 10 秒强时间标注声景，训练集、验证集和测试集按
source recording ID 完全隔离。每个声景含 2-4 个事件并允许重叠，CRNN 输出 20 ms 帧级后验，
再经过验证集阈值选择、平滑、短间隔合并和最短时长过滤。

```bash
.venv/bin/python -m pip install -e '.[sed]'
.venv/bin/python -m audioscope.cli sed-train-evaluate \
  --config configs/esc10_polyphonic_crnn.yaml
.venv/bin/python -m audioscope.cli sed-predict \
  --model artifacts/esc10-polyphonic-crnn/crnn.pt \
  --audio data/sed-esc10-polyphonic/audio/test/test-0000.wav
```

完整协议、event/segment 指标、阈值消融、逐类错误和合成标注限制见
`reports/ESC10_POLYPHONIC_CRNN.md`。

### L3 增强：真实强标注与模型对照

增强实验只使用 DESED 重打包数据中真正带 onset/offset 的 1153 条真实录音，并按源视频 ID
哈希为 703/229/221 条 train/validation/test。弱标签 train pool 不进入该实验。运行：

```bash
.venv/bin/python -m audioscope.cli sed-enhance-evaluate \
  --config configs/desed_real_enhanced.yaml

.venv/bin/python -m audioscope.cli sed-predict \
  --model artifacts/desed-real-enhanced/crnn_short_event.pt \
  --audio example.wav --threshold-profile short_event
```

artifact 支持 `overall`、`class_balanced`、`short_event` 三种验证集选择的阈值 profile。完整
真实数据协议、短事件消融及 CRNN–Conformer 精度/成本对照见
`reports/DESED_REAL_ENHANCED.md`。

## L4 工程部署

一条命令生成校准部署包、开放集评测、ONNX/INT8、长音频样例、全测试集 SNR 曲线和 CPU
性能报告：

```bash
.venv/bin/python -m pip install -e '.[engineering,demo,microphone,dev]'
.venv/bin/audioscope l4-build --config configs/l4_engineering.yaml
```

长音频与流式文件推理：

```bash
.venv/bin/audioscope sed-long-predict \
  --model artifacts/l4-engineering/crnn_short_event_calibrated.pt \
  --onnx artifacts/l4-engineering/crnn_short_event_fp32.onnx \
  --audio data/l4-engineering/l4_long_demo.wav

.venv/bin/audioscope sed-stream-file \
  --model artifacts/l4-engineering/crnn_short_event_calibrated.pt \
  --onnx artifacts/l4-engineering/crnn_short_event_fp32.onnx \
  --audio data/l4-engineering/l4_long_demo.wav
```

上传/录音演示：

```bash
.venv/bin/python l4_app.py \
  --model artifacts/l4-engineering/crnn_short_event_calibrated.pt \
  --onnx artifacts/l4-engineering/crnn_short_event_fp32.onnx
```

完整协议、真实性边界和失败量化实验见 `reports/L4_ENGINEERING.md`；运行说明见
`docs/L4_DEPLOYMENT.md`。

## 数据库、Docker 与发布完善

项目现在提供 PostgreSQL 数据血缘模型、Alembic 迁移、ESC-50/DESED/模型/指标幂等导入、
FastAPI 只读接口和 Docker Compose。音频与 checkpoint 不塞入 SQL，而以 URI 和 SHA-256 关联。
完整操作见 `docs/DATABASE_AND_DOCKER.md`。

非现场发布工作还包括 3 设备×3 房间的连续播放采集包、同步对齐与自动质检、文件流验收、
中英文摘要、简历要点和演示视频脚本。现场只需录制真实设备/房间数据、运行实体麦克风验收并
录制最终视频，详见 `docs/FIELD_OPERATIONS.md`。英文首页见 `README.en.md`。

## 复现约束

- 官方 fold 是唯一正式比较协议；同一源录音及其派生切片禁止跨 fold。
- ESC-50 官方 fold 自身存在 4 个跨 fold 的 Freesound source ID；为保持可比性保留并显式披露，
  后续另做 source-disjoint 敏感性分析。
- 所有音频必须有来源、许可、采样率、时长、校验值和 source recording ID。
- 大型数据、特征、checkpoint 与 WAV 不提交 Git。
- `reports/generated/` 中的 smoke 数字不得表述为真实数据集性能。
- 下一阶段与完成标准见 `docs/ROADMAP.md`。

## 许可

AudioScope 源码采用 [MIT License](LICENSE)。第三方数据集、音频、预训练模型及其派生成果仍受
各自许可和来源条款约束；`data/sources.yaml` 中的许可记录优先于代码许可。

## Rights and attribution / 权利与署名

Original project author/maintainer: **Crescent Feng (CrescentFeng)**. The current MIT licence retains copyright while permitting reuse under its terms, including commercial reuse; it is not an all-rights-reserved/no-commercial-use policy. Third-party material retains its own licence.

- [Copyright, attribution and contact](COPYRIGHT.md)
- [Citation](CITATION.cff) and [third-party notices](THIRD_PARTY_NOTICES.md)
- [Asset provenance](docs/ASSET_PROVENANCE.md)
- [Contributing](CONTRIBUTING.md) and [security reporting](SECURITY.md)
