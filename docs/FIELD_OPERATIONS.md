# AudioScope 现场采集与自动验收

## 已经由工程准备好的内容

运行以下命令会从 DESED test 中确定性挑选 20 条覆盖十类的真实强标注片段，统一响度，插入
同步扫频和间隔，并生成连续播放音轨、逐片段真值与 3 设备 × 3 房间会话表：

```bash
.venv/bin/python -m audioscope.field_ops prepare --root . --clips 20
```

生成目录 `data/l4-field-kit/`。参考音轨仅用于本地测试，不应因数据集许可而作为作品集附件再分发。

## 现场只做四件事

1. 在 `recording_matrix.csv` 写入真实设备名、房间名、距离和播放电平；
2. 每个设备×房间组合，从头完整播放一次 `reference_playback.wav` 并连续录音；
3. 将九个 WAV 放入表内 `recording_path` 指定的位置，保持 PCM WAV，禁止降噪和自动剪辑；
4. 把对应行的 `status` 改为 `captured`，记录异常后运行自动质检。

```bash
.venv/bin/python -m audioscope.field_ops qc --root . \
  --model artifacts/l4-engineering/crnn_short_event_calibrated.pt
```

程序会检查时长、静音、削波，估计同步偏移，并按参考时间切出逻辑片段后计算各会话的
clip-level precision/recall/F1。结果写入 `reports/generated/field/field_qc.json`。任何缺失会话都
保持 `awaiting_field_capture`，不会被误报为完成。

## 实体麦克风验收

无硬件阶段先运行文件流模拟：

```bash
.venv/bin/python -m audioscope.mic_acceptance \
  --model artifacts/l4-engineering/crnn_short_event_calibrated.pt \
  --onnx artifacts/l4-engineering/crnn_short_event_fp32.onnx \
  --simulate-file data/l4-engineering/l4_long_demo.wav \
  --output reports/generated/field/mic_simulation.json
```

接入麦克风并授权系统权限后执行真实验收：

```bash
.venv/bin/python -m audioscope.mic_acceptance \
  --model artifacts/l4-engineering/crnn_short_event_calibrated.pt \
  --onnx artifacts/l4-engineering/crnn_short_event_fp32.onnx \
  --live-seconds 300 \
  --output reports/generated/field/mic_live_acceptance.json
```

最终发布审计会区分文件模拟与硬件验收，不能用前者替代后者。
