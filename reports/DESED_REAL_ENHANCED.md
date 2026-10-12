# AudioScope L3 增强：DESED 真实强标注、短事件与 CRNN–Conformer 对照

## 结论

本轮完成了真实录音强时间标注验证、类别独立阈值、短事件专项训练/工作点，以及 CRNN 与
轻量 Conformer 的同协议对照。当前推荐模型是 `crnn_short_event.pt`：默认 overall profile
用于整体 SED；对 ≤1 秒事件更敏感时使用 short_event profile。

| 模型 / profile | Test micro event-F1 | Macro event-F1 | 1 s segment-F1 | 短事件 F1 |
|---|---:|---:|---:|---:|
| CRNN 基线 / overall | 0.258 | 0.145 | 0.558 | 0.180 |
| CRNN 短事件训练 / overall | 0.258 | **0.184** | **0.593** | 0.179 |
| **CRNN 短事件训练 / short_event** | **0.274** | — | — | **0.210** |
| Lite Conformer / overall | 0.221 | 0.178 | 0.567 | 0.154 |

短事件训练本身把短事件召回从 `0.205` 提高到 `0.360`，但也增加假阳性；再使用只在
validation 上选择的短事件阈值 `0.9` 后，测试短事件 precision/recall/F1 为
`0.160/0.305/0.210`。相对 CRNN 基线短事件 F1 提升约 16.5%。

## 1. 数据与划分

数据来自 Zenodo 的 “DESED in WebDataset Format” v2 重打包记录
（DOI `10.5281/zenodo.14808180`，记录声明 CC BY 4.0）。40 个 tar 分片均按记录公布的
MD5 校验通过。

检查后发现：`train` 池的 1578 条录音是弱标签片段，事件标签基本覆盖整段，无法支持真实
onset/offset 或短事件训练；真正的人工作强标注位于 1153 条 `eval` 池录音。因此本实验排除
弱标签池，只将强标注池按源 YouTube 视频 ID 的 SHA-256 稳定哈希切成 60%/20%/20%。

| Split | Clips | Events | ≤1 s events | Unique source IDs |
|---|---:|---:|---:|---:|
| Train | 703 | 2621 | 1264 | 703 |
| Validation | 229 | 826 | 398 | 229 |
| Test | 221 | 777 | 361 | 221 |

三组源 ID 两两交集为空。所有阈值和工作点只使用 validation；test 不参与模型、阈值或后处理
选择。

类别为 `alarm_bell_ringing`、`blender`、`cat`、`dishes`、`dog`、
`electric_shaver_toothbrush`、`frying`、`running_water`、`speech`、
`vacuum_cleaner`。类别明显不均衡，例如训练集 speech 有 1107 个事件，而
electric_shaver_toothbrush 只有 39 个。

## 2. 共用协议

- 输入：16 kHz mono，64-bin log-mel；25 ms 窗、20 ms hop、500 帧；
- CNN 前端：三层卷积，频率轴下采样；
- 损失：frame-wise BCE，正样本权重上限 10；
- 增强：时间和频率遮挡；
- 优化：AdamW，最多 20 epochs；
- 后处理：3 帧平滑、0.1 秒 gap merge、0.06 秒 minimum duration；
- event collar：onset 0.2 秒，offset `max(0.2 秒, 参考时长 20%)`；
- segment：1 秒网格；
- 短事件定义：参考或预测事件时长 ≤1 秒。

短于 10 秒的片段使用有效帧 mask，padding 区域不计入训练损失和事件解码。

## 3. 三种模型变体

### 3.1 CRNN 基线

246,762 参数。CNN 后接 64 hidden size 双向 GRU，不使用短事件额外权重。

### 3.2 CRNN 短事件训练

结构与基线完全一致，保持相同参数量。对 ≤1 秒正事件帧赋予 2.5 倍权重，对 onset/offset
附近 80 ms 区域赋予 1.5 倍权重，用于提高短事件与边界的梯度贡献。

### 3.3 Lite Conformer 短事件训练

184,362 参数。CNN 后使用 2 层、64 维、4-head Conformer，内部时间轴降到 125 帧，最后
插值回 500 帧输出。使用与增强 CRNN 相同的短事件损失。

| 变体 | 参数 | CPU 训练 | 221 条测试推理* |
|---|---:|---:|---:|
| CRNN 基线 | 246,762 | 183.5 s | 0.66 s |
| CRNN 短事件 | 246,762 | 182.9 s | 0.65 s |
| Lite Conformer | 184,362 | 592.2 s | 2.35 s |

\* 只统计已缓存 log-mel 的模型前向，不是端到端实时系数。

## 4. 阈值策略

### 4.1 Overall profile

在 validation 上最大化 micro event-F1。CRNN 基线选择 `0.9`，增强 CRNN 和 Conformer
选择 `0.8`。这是模型 artifact 的默认 profile。

### 4.2 Class-balanced profile

每类独立扫描阈值，以该类 validation event-F1 为主、segment-F1 为次。实验表明它不是
稳定的 micro 提升方式：

| 模型 | Global micro / macro | Per-class micro / macro |
|---|---:|---:|
| CRNN 基线 | 0.258 / 0.145 | 0.237 / **0.178** |
| CRNN 短事件 | **0.258 / 0.184** | 0.228 / 0.167 |
| Lite Conformer | **0.221 / 0.178** | 0.208 / 0.170 |

逐类阈值在基线中改善 macro、牺牲 micro；在另外两种模型中 test macro 也下降，说明小样本
类别的 validation 阈值有过拟合。该 profile 保留给类别均衡分析，不作为默认部署策略。

### 4.3 Short-event profile

使用 validation 的 ≤1 秒 event-F1 选择单一阈值。增强 CRNN 从 overall 的 `0.8` 提高到
`0.9`：

| 指标 | CRNN 基线 overall | 增强 CRNN overall | 增强 CRNN short_event |
|---|---:|---:|---:|
| Short-event precision | 0.161 | 0.119 | 0.160 |
| Short-event recall | 0.205 | **0.360** | 0.305 |
| Short-event F1 | 0.180 | 0.179 | **0.210** |
| Overall event-F1 | 0.258 | 0.258 | **0.274** |

结果说明加权损失主要把工作点推向高召回；validation 选择的更高阈值重新平衡 precision，才
真正形成短事件 F1 增益。

## 5. CRNN 与 Conformer

本预算下 Lite Conformer 没有超过 CRNN：event-F1 `0.221` 对 `0.258`，segment-F1
`0.567` 对 `0.593`，训练约慢 3.2 倍，缓存特征推理约慢 3.6 倍。结论只适用于当前
703 条训练录音、轻量模型和纯监督训练，不代表更大规模预训练 Conformer 的上限。

当前保留 CRNN 作为 L3 默认模型。若继续投入 Conformer，应优先测试预训练表示、更多真实强
标注、学习率 warmup 和更长训练，而不是直接放大当前随机初始化模型。

## 6. 逐类观察

增强 CRNN overall profile 的 event-F1 前两类为 speech `0.417`、alarm_bell_ringing
`0.385`；最弱为 frying `0.034`、running_water `0.085`、
electric_shaver_toothbrush `0.085`。1 秒 segment-F1 对 speech 为 `0.784`，对
alarm_bell_ringing 为 `0.771`，说明模型对粗粒度活动区间的识别远好于严格边界切分。

稀有类别事件数很小，单次 source-hash split 的逐类结论方差会较高。下一轮应采用多 split
重复或 group cross-validation，并为 rare classes 设计采样策略。

## 7. 使用

训练和完整对照：

```bash
.venv/bin/python -m audioscope.cli sed-enhance-evaluate \
  --config configs/desed_real_enhanced.yaml
```

三种阈值 profile：

```bash
# 整体 event-F1 工作点（默认）
.venv/bin/python -m audioscope.cli sed-predict \
  --model artifacts/desed-real-enhanced/crnn_short_event.pt \
  --audio example.wav --threshold-profile overall

# 类别独立阈值
.venv/bin/python -m audioscope.cli sed-predict \
  --model artifacts/desed-real-enhanced/crnn_short_event.pt \
  --audio example.wav --threshold-profile class_balanced

# 短事件工作点
.venv/bin/python -m audioscope.cli sed-predict \
  --model artifacts/desed-real-enhanced/crnn_short_event.pt \
  --audio example.wav --threshold-profile short_event
```

## 8. 边界

1. 数据源是第三方 Zenodo WebDataset 重打包；许可说明取自该记录，使用者仍需核对底层
   AudioSet/YouTube 内容条款；
2. 强标注池只有 1153 条，且类别不均衡；
3. 本实验是单次确定性 group split，不是官方 challenge 排名协议；
4. Conformer 是 CPU 预算下的轻量随机初始化版本；
5. 短事件 profile 是不同工作点，不应与 overall profile 混用后再挑 test 最优结果；
6. 推理耗时不包含特征提取，端到端实时性仍属于 L4。
