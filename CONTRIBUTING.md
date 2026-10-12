# Contributing

感谢你改进 AudioScope。请先创建 issue 说明问题、实验假设或界面变更，再提交小而聚焦的 PR。

## 开发环境

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[engineering,database,api,demo,dev]'
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest
```

不要提交 `.env`、用户音频、数据库内容、第三方数据集、预训练权重、训练 checkpoint 或缓存。
新增数据源必须在 `data/sources.yaml` 登记来源、许可、版本和用途限制。指标变更应同时说明 split、
阈值选择集、随机种子与失败结果；真实现场验证和软件模拟必须明确区分。
