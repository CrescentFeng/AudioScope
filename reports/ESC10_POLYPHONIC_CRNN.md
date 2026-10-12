# AudioScope L3：ESC-10 复音事件检测 CRNN 基线

## 结论

AudioScope 已完成 L3 的第一版可复现实验闭环：从 ESC-10 源片段生成带 onset/offset
强时间标注的重叠声景，训练帧级多标签 CRNN，在验证集选择阈值，并在隔离测试集报告
event-F1 与 segment-F1。该结果是受控合成复音协议下的基线，不代表真实录音域上的事件检测能力。

| 测试指标 | Precision | Recall | F1 |
|---|---:|---:|---:|
| Event-based micro | 0.224 | 0.276 | **0.248** |
| Segment-based micro（1 秒） | 0.525 | 0.595 | **0.557** |

测试集 macro event-F1 为 `0.241`，macro segment-F1 为 `0.539`。

## 1. 与项目规划的对应关系

本阶段覆盖规划中的四个核心要素：

1. 多标签帧级预测，可同时输出重叠事件；
2. onset/offset 时间定位；
3. CRNN 时序模型；
4. 阈值、平滑、短间隔合并、最短事件时长，以及 event/segment 两类 F1。

这次先实现 CRNN 作为可复现基线，Conformer、真实强标注数据和在线流式检测留给后续迭代。

## 2. 数据协议

### 2.1 受控声景生成

- 源数据：固定版本 ESC-50 中的十个 ESC-10 类别；
- 类别：`chainsaw`、`clock_tick`、`crackling_fire`、`crying_baby`、`dog`、
  `helicopter`、`rain`、`rooster`、`sea_waves`、`sneezing`；
- 声景：16 kHz、单声道、每条 10 秒；
- 每条包含 2–4 个事件，事件裁剪长度 1–3 秒；
- 以随机 onset、增益和背景噪声进行混合，允许事件重叠；
- 随机种子固定为 `2027`。

| Split | Scenes | Events | Unique source IDs | 含重叠事件的场景 |
|---|---:|---:|---:|---:|
| Train | 300 | 921 | 185 | 275（91.7%） |
| Validation | 100 | 301 | 58 | 91（91.0%） |
| Test | 100 | 293 | 55 | 88（88.0%） |
| 合计 | 500 | 1515 | — | 454 |

实测最大同时发声数为 4。

### 2.2 泄漏控制

训练使用 ESC folds 1–3，验证使用 fold 4，测试使用 fold 5。生成前按源录音 ID 检查边界，
并移除跨 split 的两个源 ID：`esc50-source-209698` 和 `esc50-source-234879`。最终三组
两两交集均为空。阈值及后处理参数只根据验证集选择，测试集不参与选择。

### 2.3 标签限制

onset/offset 是混音计划的精确区间，但 ESC-50 原始标签是片段级标签。若某个裁剪片段内部有
静音或间歇，标签仍把整个裁剪区间视为事件活跃，因此会高估真实声学活动区间。源片段也可能在
同一 split 内复用。所有分数必须表述为“合成复音协议结果”。

## 3. 模型与训练

输入是 64-bin log-mel，25 ms 分析窗、20 ms hop，每条声景形成 `64 × 500` 特征。
三层卷积只压缩频率轴，保留 500 帧时间轴；随后使用双向 GRU，为十个类别逐帧输出独立
sigmoid 概率。总参数量为 `246,762`。

- 损失：frame-wise BCE，正样本权重上限 `10`；
- 优化器：AdamW，学习率 `1e-3`，weight decay `1e-4`；
- 增强：频率遮挡与时间遮挡；
- batch size：8；
- 最多 30 epochs，验证损失 early stopping；
- 实际完成 29 epochs，最佳 checkpoint 为 epoch 22；
- 最佳验证损失：`0.480876`；
- 本机 CPU 训练耗时：`137.3 s`。

## 4. 后处理与评测

最终配置使用：

- 全局阈值：`0.8`；
- 5 帧概率平滑；
- 相邻事件间隔不超过 `0.6 s` 时合并；
- 丢弃短于 `0.4 s` 的事件；
- event 匹配：onset collar `0.2 s`，offset 容差为 `max(0.2 s, 参考事件时长的 20%)`；
- segment 指标：1 秒时间网格。

Event-based 指标要求同类事件的开始和结束都落在容差内；segment-based 指标只判断每个
固定时间段中类别是否活跃。因此 segment-F1 明显高于 event-F1，说明模型较能识别“何时大致
有声音”，但精确切分事件边界仍是主要短板。

### 4.1 验证集阈值扫描

| Threshold | Validation event-F1 | Validation segment-F1 |
|---:|---:|---:|
| 0.700 | 0.255 | 0.546 |
| **0.800** | **0.277** | **0.574** |
| 0.850 | 0.277 | 0.567 |
| 0.900 | 0.234 | 0.530 |
| 0.925 | 0.202 | 0.473 |
| 0.950 | 0.165 | 0.403 |
| 0.975 | 0.060 | 0.243 |

按验证集 event-F1 选择 `0.8`，之后冻结配置再计算测试结果。

### 4.2 最终测试结果

| 指标 | Micro F1 | Macro F1 |
|---|---:|---:|
| Event-based | **0.248** | 0.241 |
| Segment-based（1 秒） | **0.557** | 0.539 |

Event-based micro 计数为 TP `81`、FP `280`、FN `212`。1 秒 segment-based 计数为
TP `512`、FP `464`、FN `349`。

### 4.3 逐类结果

| 类别 | Event-F1 | Segment-F1 |
|---|---:|---:|
| chainsaw | 0.373 | 0.762 |
| clock_tick | 0.034 | 0.376 |
| crackling_fire | 0.290 | 0.549 |
| crying_baby | 0.238 | 0.619 |
| dog | 0.225 | 0.577 |
| helicopter | 0.282 | 0.426 |
| rain | 0.453 | 0.747 |
| rooster | 0.051 | 0.455 |
| sea_waves | **0.462** | 0.743 |
| sneezing | 0.000 | 0.133 |

`sea_waves`、`rain`、`chainsaw` 的持续纹理较容易稳定识别；`sneezing`、`clock_tick`、
`rooster` 的短促或内部间歇特征导致边界匹配很弱。下一轮应优先使用更精确的真实强标注、
类别独立阈值，以及针对短事件的采样与损失设计。

### 4.4 单条时间轴检查

`test-0000` 中，模型准确匹配 `helicopter`（参考 5.856–6.899 秒，预测 5.860–6.900 秒）
与后段 `chainsaw`（参考 8.687–9.970 秒，预测 8.760–9.840 秒），但把 3.254–4.604 秒的
`rooster` 错判为 `chainsaw`，并在末尾产生一个重叠的 `sea_waves` 假阳性。该例同时暴露出
类别混淆和事件边界问题。

## 5. 训练修正记录

首轮模型使用未截断的正样本权重（约 14.7–19.1），最佳验证损失为 `0.612433`；最终模型把
权重上限设为 10，最佳验证损失降至 `0.480876`。首轮原始后处理下测试 event-F1/segment-F1
为 `0.204/0.534`，最终完整配置为 `0.248/0.557`。由于正样本权重和后处理同时发生调整，
不能把全部增益单独归因于权重截断；首轮 checkpoint 与报告保留在 `ablations/` 供复查。

## 6. 复现

```bash
.venv/bin/python -m pip install -e '.[sed]'
.venv/bin/python -m audioscope.cli sed-train-evaluate \
  --config configs/esc10_polyphonic_crnn.yaml

.venv/bin/python -m audioscope.cli sed-predict \
  --model artifacts/esc10-polyphonic-crnn/crnn.pt \
  --audio data/sed-esc10-polyphonic/audio/test/test-0000.wav
```

关键产物：

- 数据与强标注：`data/sed-esc10-polyphonic/`；
- 特征与帧标签缓存：`data/processed/esc10-polyphonic-sed/`；
- 最终模型：`artifacts/esc10-polyphonic-crnn/crnn.pt`；
- 汇总与逐条预测：`reports/generated/esc10-polyphonic-crnn/`；
- 数据卡与模型卡：`docs/SED_DATASET_CARD.md`、`docs/SED_MODEL_CARD.md`。

## 7. 本阶段边界与下一步

本次完成的是 L3 v1 基线闭环，不是整个真实域 L3 的终点。仍需：

1. 在真实强标注声景上训练与独立评测；
2. 加入 per-class threshold、事件边界损失或 Conformer 对照；
3. 处理 unknown/open-set 事件与置信度校准；
4. 扩充短促事件并进行多随机种子或交叉验证；
5. 在 L4 中测量端到端特征提取、模型推理和流式后处理的实时系数。
