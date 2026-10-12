# AudioScope 数据库与 Docker 运行手册

## 数据边界

PostgreSQL 保存来源、许可、数据集版本、文件 URI、SHA-256、强/弱标注、切分、处理运行、模型
产物、基准指标和现场验收记录。WAV、NPY、NPZ、PT 与 ONNX 不写入数据库；它们保留在文件系统
或对象存储，数据库只记录可迁移 URI 与完整性哈希。这避免数据库膨胀，也保留了数据血缘。

数据库采用独立 Compose 项目 `audioscope`，默认监听宿主机 `55432`，不会复用或覆盖其它
PostgreSQL 容器。初始化和升级只通过 Alembic 执行。

## 首次启动

```bash
cp .env.example .env
# 编辑 .env，把 AUDIOSCOPE_DB_PASSWORD 换成长随机密码；不要提交 .env。
docker compose up -d db
docker compose run --rm migrate

AUDIOSCOPE_DATABASE_URL='postgresql+psycopg://audioscope:密码@127.0.0.1:55432/audioscope' \
  .venv/bin/python -m audioscope.catalog_import \
  --root . --hash-files --output reports/generated/database/import_report.json

docker compose up -d api
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/readiness
```

`--hash-files` 会实际读取 3,153 个原始 WAV。重复执行导入是幂等的：稳定主键会更新相同实体，
不会复制数据。导入报告只有在数据集、音频、标注、切分、模型和指标达到最低预期数量时才通过。

## 服务

| 服务 | 用途 | 默认端口/行为 |
|---|---|---|
| `db` | PostgreSQL 16 元数据目录 | `55432` |
| `migrate` | 一次性 Alembic 升级 | 成功后退出 |
| `api` | 只读目录与指标 API | `8000` |
| `demo` | Gradio L4 推理界面 | profile 启动，`7860` |

启动演示界面：

```bash
docker compose --profile demo up -d demo
```

## 日常检查与备份

```bash
docker compose ps
docker compose logs --tail=100 api
docker compose exec -T db pg_dump -U audioscope -d audioscope -Fc > audioscope.dump
```

恢复前应新建空数据库并先验证备份清单，不要覆盖唯一实例。密码轮换后同步更新 `.env`，重建
`db` 以外的客户端服务。任何出现在 shell 历史或日志中的旧口令都应视为已泄露。

## API

- `GET /health`：进程、数据库连通性与迁移版本；
- `GET /v1/readiness`：关键实体计数与导入就绪状态；
- `GET /v1/datasets`：数据集版本、manifest 哈希和记录数；
- `GET /v1/models`：模型 URI、格式、大小和 SHA-256；
- `GET /v1/benchmarks`：CPU 延迟、RTF、鲁棒性等机器可读指标。

OpenAPI 页面位于 `http://127.0.0.1:8000/docs`。

如果构建机暂时无法访问 Python 基础镜像仓库，可用本机缓存的 Adminer/PHP 镜像启动只读兼容
接口进行离线验收：`docker compose --profile offline up -d api-offline`。它在 `8001` 暴露相同的
健康、就绪、数据集、模型和基准路径，但不提供 OpenAPI；正式服务仍是 FastAPI `api`。
