"""A bounded, idempotent local launcher; opens the browser only after readiness."""

import argparse
import fcntl
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:7862/"
PORT = 7862


class StartupError(RuntimeError):
    pass


def local_environment():
    env = os.environ.copy()
    env["AUDIOSCOPE_DEMO_PORT"] = str(PORT)
    for key in ("NO_PROXY", "no_proxy"):
        old = [p.strip() for p in env.get(key, "").split(",") if p.strip()]
        env[key] = ",".join(dict.fromkeys([*old, "127.0.0.1", "localhost", "::1"]))
    return env


def find_docker():
    candidates = [
        shutil.which("docker"),
        "/usr/local/bin/docker",
        "/opt/homebrew/bin/docker",
        "/Applications/Docker.app/Contents/Resources/bin/docker",
    ]
    for value in candidates:
        if value and Path(value).is_file() and os.access(value, os.X_OK):
            return str(value)
    raise StartupError("没有找到 Docker Desktop。请先安装并完成首次启动设置；不会自动安装软件。")


def owns_port(details):
    return any(
        p.get("HostIp") == "127.0.0.1" and p.get("HostPort") == str(PORT)
        for p in details.get("NetworkSettings", {}).get("Ports", {}).get("7860/tcp") or []
    )


def port_busy():
    with socket.socket() as probe:
        probe.settimeout(1)
        return probe.connect_ex(("127.0.0.1", PORT)) == 0


class Launcher:
    def __init__(self, root=ROOT):
        self.root = root
        self.env = local_environment()
        self.docker = find_docker()
        self.messages = []
        self.report = {
            "status": "starting",
            "url": URL,
            "started_at": datetime.now().astimezone().isoformat(),
        }

    def say(self, text):
        self.messages.append(text)
        print(text, flush=True)

    def command(self, args, timeout=15, required=True):
        try:
            result = subprocess.run(
                [self.docker, *args],
                cwd=self.root,
                env=self.env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            if required:
                raise StartupError(
                    "服务命令等待超时。请检查 Docker Desktop 是否已完成启动，再重试。"
                ) from None
            return None
        if result.returncode:
            if required:
                # Do not display raw Docker/DB output or credential-bearing configuration.
                raise StartupError(
                    f"Docker 的 {args[0]} 操作未完成。"
                    "请检查 Docker Desktop 的 AudioScope 容器状态。"
                )
            return None
        return result.stdout.strip()

    def compose(self, *args, timeout=30):
        return self.command(
            [
                "compose",
                "--project-name",
                "audioscope",
                "-f",
                str(self.root / "compose.yaml"),
                "--profile",
                "demo",
                *args,
            ],
            timeout=timeout,
        )

    def container(self, service):
        value = self.compose("ps", "-a", "-q", service)
        if not value:
            return None
        ids = value.splitlines()
        if len(ids) != 1:
            raise StartupError(f"发现多个 {service} 容器，停止自动操作，避免启动错误服务。")
        return json.loads(self.command(["inspect", ids[0]]))[0]

    def wait(self, predicate, seconds, message):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if predicate():
                return True
            self.say(message)
            time.sleep(3)
        return False

    def daemon(self, start):
        host = self.env.get("DOCKER_HOST", "")
        if host and not host.startswith("unix://"):
            raise StartupError("Docker 指向远程主机，停止操作。请先手动切换回本机 Docker Desktop。")
        endpoint = self.command(["context", "inspect", "--format", "{{.Endpoints.docker.Host}}"])
        if not endpoint.startswith("unix://"):
            raise StartupError(
                "当前 Docker context 不是本机 Unix socket，不自动切换或操作远程服务。"
            )

        def available():
            return (
                self.command(["info", "--format", "{{.ServerVersion}}"], timeout=5, required=False)
                is not None
            )

        if available():
            return
        if not start:
            raise StartupError("Docker 尚未启动。")
        self.say("正在启动 Docker Desktop，首次启动可能需要一两分钟……")
        self.command(["desktop", "start", "--detach", "--timeout", "10"])
        if not self.wait(available, 150, "等待 Docker 就绪……"):
            raise StartupError(
                "Docker 未在规定时间内就绪。"
                "请打开 Docker Desktop 完成可能出现的权限/首次设置提示后重试。"
            )

    def ready(self, details, inference=False):
        if not details or not details["State"].get("Running") or not owns_port(details):
            return False
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(URL + "config", timeout=3) as response:
                config = json.load(response)
            if not str(config.get("title", "")).startswith("AudioScope"):
                return False
        except (OSError, ValueError):
            return False
        args = ["exec", details["Id"], "python", "-m", "audioscope.workbench_health"]
        if inference:
            args.append("--inference")
        output = self.command(args, timeout=45 if inference else 15, required=False)
        if not output:
            return False
        try:
            health = json.loads(output)
        except ValueError:
            return False
        good = health.get("status") == "ready" and health.get("page") and health.get("database")
        if inference:
            good = good and health.get("inference_tested")
        if good:
            self.report["checks"] = health
        return bool(good)

    def ensure(self, check_only=False):
        self.say("正在检查 AudioScope 的固定入口、Docker、数据库与模型……")
        for required in (".env", "compose.yaml", "artifacts/workbench-release/model_registry.json"):
            if not (self.root / required).is_file():
                raise StartupError(
                    f"项目缺少 {required}；请恢复项目文件，不会创建空白历史或下载替代模型。"
                )
        self.daemon(start=not check_only)
        current = self.container("demo")
        if self.ready(current, inference=True):
            self.say("AudioScope 已在运行，页面、数据库和模型检查通过。")
            return
        if check_only:
            raise StartupError("AudioScope 尚未通过完整就绪检查；检查模式未启动或重启任何服务。")
        if port_busy() and not (current and owns_port(current)):
            raise StartupError(
                "7862 端口被其他程序占用。不会关闭其他程序或跳到别的端口；请释放此端口后重试。"
            )
        self.command(["image", "inspect", "audioscope-python:local"])
        db = self.container("db")
        if db is None:
            # Existing project's data volume must survive container recreation.
            if (
                self.command(
                    ["volume", "inspect", "audioscope_audioscope_postgres_data"], required=False
                )
                is None
            ):
                raise StartupError(
                    "原 PostgreSQL 数据卷不存在；为避免空白历史，不会自动新建数据库。"
                )
            self.compose("up", "-d", "--no-deps", "--no-build", "db")
        elif not db["State"].get("Running"):
            self.say("正在恢复 AudioScope 数据库……")
            self.command(["start", db["Id"]])

        def database_ready():
            db = self.container("db")
            return db and db["State"].get("Health", {}).get("Status") == "healthy"

        if not self.wait(database_ready, 60, "等待数据库健康检查……"):
            raise StartupError("数据库未就绪。请检查 Docker 的 db 容器，不会删除或重置数据。")
        self.say("正在启动 AudioScope 页面服务……")
        self.compose("up", "-d", "--no-deps", "--no-build", "demo")
        if not self.wait(
            lambda: self.ready(self.container("demo")), 75, "等待页面和历史记录接口就绪……"
        ):
            self.say("页面仍未就绪，只对 AudioScope 页面服务重启一次……")
            self.compose("restart", "demo")
            if not self.wait(lambda: self.ready(self.container("demo")), 60, "重新检查页面……"):
                raise StartupError(
                    "页面仍无法就绪，已停止自动重试。请检查模型挂载和容器日志；未改动数据库。"
                )
        self.say("正在核验模型文件并做一次不保存数据的推理自检……")
        if not self.ready(self.container("demo"), inference=True):
            raise StartupError("模型自检未通过；不会打开看似正常但无法分析的页面，也不会替换模型。")

    def save_report(self):
        folder = self.root / "runtime/startup"
        folder.mkdir(parents=True, exist_ok=True)
        self.report["messages"] = self.messages
        self.report["finished_at"] = datetime.now().astimezone().isoformat()
        target = folder / "latest.json"
        temporary = folder / "latest.json.tmp"
        temporary.write_text(json.dumps(self.report, ensure_ascii=False, indent=2) + "\n")
        temporary.chmod(0o600)
        temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-open", action="store_true", help="只启动并验证，不打开浏览器")
    parser.add_argument(
        "--check", action="store_true", help="只读诊断，不启动/重启服务或打开浏览器"
    )
    args = parser.parse_args()
    launcher = None
    folder = ROOT / "runtime/startup"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "launch.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("另一个 AudioScope 启动检查正在进行，请等待它完成；不会重复启动。")
            return 0
        try:
            launcher = Launcher()
            launcher.ensure(check_only=args.check)
            launcher.report["status"] = "ready"
            launcher.say("已就绪：" + URL)
            if not args.no_open and not args.check:
                subprocess.run(["/usr/bin/open", URL], check=True, timeout=10)
        except (StartupError, OSError, subprocess.SubprocessError) as exc:
            message = (
                str(exc)
                if isinstance(exc, StartupError)
                else "启动系统操作未完成，请检查 Docker Desktop 后重试。"
            )
            print("未打开网页：" + message, file=sys.stderr)
            if launcher:
                launcher.report.update(status="failed", error=message)
            return 1
        finally:
            if launcher:
                launcher.save_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
