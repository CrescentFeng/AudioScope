# AudioScope L4 deployment model card

## Intended use

AudioScope L4 detects ten domestic sound-event classes in mono audio: alarm bell ringing, blender,
cat, dishes, dog, electric shaver/toothbrush, frying, running water, speech and vacuum cleaner. It is
designed for portfolio demonstrations, offline long-file analysis and low-latency CPU prototypes.

## Not intended for

It is not a safety alarm, medical monitor, surveillance decision system or general-purpose acoustic
classifier. `unknown` means low confidence under a simple rejection baseline; it does not identify
or guarantee detection of every out-of-domain sound.

## Training and evaluation data

The model was trained on the strong-label pool of the DESED WebDataset repack and evaluated on a
source-disjoint deterministic test split. Open-set calibration uses ESC-50 classes outside the ten
deployed labels. Review the dataset card and source registry before redistributing audio or weights.

## Performance summary

- clean DESED test: event-F1 0.258, 1-second segment-F1 0.585;
- 10 dB white noise: event-F1 0.155, segment-F1 0.390;
- open-set AUROC 0.750 with 0.553 unknown recall at the selected threshold;
- ONNX FP32 10-second window P50/P95: 0.873/0.967 ms on the measured Apple CPU;
- PyTorch–ONNX maximum absolute logit difference: 2.38e-6.

Hardware timings are not portable. The open-set and noisy-condition results show that the model is
not reliable enough for consequential decisions.

## Ethical and privacy notes

The speech label only reports acoustic activity and performs no transcription or speaker identity
inference. Microphone users must obtain appropriate consent and comply with local recording laws.
Audio is processed locally by the supplied demo unless the operator modifies its hosting setup.

## Known limitations

- severe degradation below 10 dB SNR;
- class imbalance and one deterministic group split;
- no room/device/real-noise domain study yet;
- low-confidence rejection remains overconfident on many ESC-50 sounds;
- dynamic INT8 does not quantize the CRNN GRU/Conv graph in the tested runtime.
# v1.0 软件验证版发布说明

AudioScope v1.0 Software-Validated Release。软件验证完成；未进行自有物理设备/房间认证。
Physical device and room certification has not been performed.

2026-10-11 的软件封板新增全量 221 条测试音频、七种确定性域偏移和重复推理校验，以及有界
增量解码和 1,800 秒加速文件流压力测试。模拟脉冲响应、带宽和非线性不对应某一实体设备；
混响尾部裁至原长度，保留源事件真值时间。指标变差也完整报告，不根据测试结果重新调阈值。
RTF、RSS 和延迟依赖运行主机与负载，不能直接推广至其它设备。首轮与全量评测并行的压力
运行曾超过内存增长预算；隔离负载后的复测通过，详见软件发布报告。

现场采集是可选的未来增强；`software-final` 验证软件版，`field-final` 和旧 `final` 仍要求
现场证据。原 v1.0 封板时正式 Python Docker 镜像受仓库网络限制，彼时已验证的是数据库容器、
兼容 API 容器和本机 FastAPI。后续工作台增强已完成正式 Python 镜像构建与真实容器工作流验收，
两次验收不是同一范围，不能用后者改写旧封板的历史记录。

## Post-v1.0 workbench enhancement (2026-10-11)

The registered workbench default retains the original neural weights and ONNX graph, with a
validation-selected event decoder. On the existing 221-clip test split, clean micro event-F1 changes
from 0.2583 to 0.2873 and macro event-F1 from 0.1865 to 0.2095. Precision improves from 0.1924 to 0.2372,
while recall decreases from 0.3925 to 0.3642. It is not an across-the-board improvement. The mean micro
event-F1 over six controlled software shifts changes from 0.2210 to 0.2568. These are software simulations,
not real room/device certification, and the previously reported test split is not a new blind benchmark.

Three-seed augmentation experiments are retained as negative evidence: their clean-validation loss
exceeded the predeclared tolerance, so their neural weights were not deployed. The original v1.0 model
remains selectable. Full protocols, duration slices and per-class errors: `reports/WORKBENCH_IMPROVEMENT.md`.

The workbench provides immutable predictions, append-only review versions, reference comparison and
local English CLAP candidate retrieval. Review is not automatically training data. CLAP similarity is
not a probability; the fixed ten-class benchmark does not establish unrestricted semantic recognition.
No abstention is implemented for absent sounds. The new API is trusted-local-single-user only, without
public authentication. Upload retention, limits and commands are documented in `docs/WORKBENCH_GUIDE.md`.
