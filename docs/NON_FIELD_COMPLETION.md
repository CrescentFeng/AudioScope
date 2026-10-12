# AudioScope 非现场工作完成报告

> 2026-10-11 更新：本页保留 9 月历史验收。正式软件完成标准已改为 `software-final`，详见
> `SOFTWARE_RELEASE.md`。原 `final` 是 `field-final` 的兼容别名，现场采集属于未来可选增强。
> 下文的“全部完成”仅表示当时已检查的范围，不构成正式 Python Docker 镜像完成构建的证据。

日期：2026-09-24

## 结论

除真实设备/房间录音、实体麦克风连续运行和最终演示视频录制外，项目的开发、数据、部署、
采集准备和发布文案均已完成。`release_audit --mode preflight` 是非现场完成门禁；
`--mode final` 会继续阻止发布，直到三项现场证据真实存在。

## 已完成且实际验证

- Alembic revision `0001_audioscope_catalog` 已应用到独立 PostgreSQL 16 容器；
- 已实际计算并导入 3,153 条 WAV 的 SHA-256；目录中有 2 个数据集版本、6,224 条标注、
  3,153 条切分、41 个模型产物、57 条基准指标和 9 个计划现场会话；
- 导入连续执行多次后计数不增长，幂等性通过；
- FastAPI 本机链路与 Docker 兼容 API 链路均返回当前迁移版本和真实目录计数；
- 已生成 239.453 秒同步播放音轨、20 条参考片段真值和 9 会话矩阵，现场录制后可自动对齐、
  质检并计算设备/房间差异；
- 119.361 秒文件流验收通过：ONNX Runtime 实时因子 0.00764、计算 P95 4.02 ms、无重复事件；
- 中英文摘要、简历要点、视频逐镜脚本、数据库/Docker 运维手册和现场操作手册已完成；
- 29 项自动测试与 Ruff 静态检查通过；发布关键产物 SHA-256 已冻结到机器可读审计报告。

## 现场完成后只需执行

```bash
.venv/bin/python -m audioscope.field_ops qc --root . \
  --model artifacts/l4-engineering/crnn_short_event_calibrated.pt

.venv/bin/python -m audioscope.mic_acceptance \
  --model artifacts/l4-engineering/crnn_short_event_calibrated.pt \
  --onnx artifacts/l4-engineering/crnn_short_event_fp32.onnx \
  --live-seconds 300 --output reports/generated/field/mic_live_acceptance.json

.venv/bin/python -m audioscope.release_audit --root . --mode final
```

最终视频保存到 `release/video/*.mp4`。审计不接受空文件、计划状态或文件流模拟冒充现场证据。

## 容器说明

本机 Docker Hub 与公共 ECR 在验收时均发生 registry metadata timeout，因此正式 FastAPI 镜像的
Python 基础层未能在线拉取。为不留下 Docker 网络/数据库链路的未验证项，已用本机缓存的
Adminer/PHP 镜像离线构建只读兼容 API 并通过真实数据库验收；正式 `Dockerfile`、FastAPI 服务
和 Compose 定义仍保留，网络恢复后运行 `docker compose build api && docker compose up -d api`
即可切回正式服务。这是构建环境的外部网络限制，不是项目代码或现场操作依赖。
