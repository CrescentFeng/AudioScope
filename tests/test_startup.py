import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from audioscope import workbench_health

spec = importlib.util.spec_from_file_location(
    "open_audioscope", Path(__file__).resolve().parents[1] / "scripts/open_audioscope.py"
)
startup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(startup)


def test_fixed_port_and_no_proxy_preserve_existing_values(monkeypatch):
    monkeypatch.setenv("AUDIOSCOPE_DEMO_PORT", "7860")
    monkeypatch.setenv("NO_PROXY", "example.com,localhost")
    env = startup.local_environment()
    assert env["AUDIOSCOPE_DEMO_PORT"] == "7862"
    assert set(env["NO_PROXY"].split(",")) >= {"example.com", "localhost", "127.0.0.1", "::1"}


@pytest.mark.parametrize(
    "bindings,expected",
    [
        (None, False),
        ([], False),
        ([{"HostIp": "0.0.0.0", "HostPort": "7862"}], False),
        ([{"HostIp": "127.0.0.1", "HostPort": "7860"}], False),
        ([{"HostIp": "127.0.0.1", "HostPort": "7862"}], True),
    ],
)
def test_port_ownership(bindings, expected):
    assert startup.owns_port({"NetworkSettings": {"Ports": {"7860/tcp": bindings}}}) is expected


class FakeLauncher(startup.Launcher):
    def __init__(self, root, running=False):
        self.root, self.messages, self.report, self.calls = root, [], {}, []
        self.running = running
        self.db_running = running

    def daemon(self, start):
        self.calls.append(("daemon", start))

    def container(self, service):
        return {
            "Id": service,
            "State": {
                "Running": self.db_running if service == "db" else self.running,
                "Health": {"Status": "healthy" if self.db_running else "starting"},
            },
            "NetworkSettings": {
                "Ports": {
                    "7860/tcp": [{"HostIp": "127.0.0.1", "HostPort": "7862"}]
                    if self.running
                    else []
                }
            },
        }

    def ready(self, details, inference=False):
        return self.running and self.db_running

    def command(self, args, **kwargs):
        self.calls.append(tuple(args))
        if args[0] == "start":
            self.db_running = True
        return "ok"

    def compose(self, *args, **kwargs):
        self.calls.append(args)
        if args[0] == "up" and args[-1] == "demo":
            self.running = True
        return "ok"

    def wait(self, predicate, seconds, message):
        return bool(predicate())


@pytest.fixture
def root(tmp_path):
    for name in (".env", "compose.yaml", "artifacts/workbench-release/model_registry.json"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    return tmp_path


def test_healthy_reopen_does_not_restart(root):
    launch = FakeLauncher(root, running=True)
    launch.ensure()
    assert launch.calls == [("daemon", True)]


def test_stopped_services_recover_without_other_apps(root, monkeypatch):
    monkeypatch.setattr(startup, "port_busy", lambda: False)
    launch = FakeLauncher(root)
    launch.ensure()
    assert ("start", "db") in launch.calls
    assert ("up", "-d", "--no-deps", "--no-build", "demo") in launch.calls
    assert not any("api" in call or "down" in call or "migrate" in call for call in launch.calls)


def test_foreign_port_is_not_killed_or_silently_changed(root, monkeypatch):
    monkeypatch.setattr(startup, "port_busy", lambda: True)
    launch = FakeLauncher(root)
    with pytest.raises(startup.StartupError, match="7862.*其他程序"):
        launch.ensure()
    assert launch.calls == [("daemon", True)]


def test_check_mode_is_read_only(root):
    launch = FakeLauncher(root)
    with pytest.raises(startup.StartupError, match="检查模式"):
        launch.ensure(check_only=True)
    assert launch.calls == [("daemon", False)]


def test_remote_docker_refused_before_any_command(monkeypatch):
    launch = startup.Launcher.__new__(startup.Launcher)
    launch.env = {"DOCKER_HOST": "tcp://example.com:2376"}
    launch.command = Mock()
    with pytest.raises(startup.StartupError, match="远程"):
        launch.daemon(True)
    launch.command.assert_not_called()


def test_docker_cold_start_is_bounded_and_explicit():
    launch = startup.Launcher.__new__(startup.Launcher)
    launch.env, launch.messages = {}, []
    launch.command = Mock(side_effect=["unix:///tmp/docker.sock", None, ""])
    launch.wait = Mock(return_value=True)
    launch.daemon(True)
    assert launch.command.call_args.args[0] == ["desktop", "start", "--detach", "--timeout", "10"]
    assert launch.wait.call_args.args[1] == 150


def test_never_open_browser_on_readiness_failure(root, monkeypatch):
    monkeypatch.setattr(startup, "ROOT", root)
    launch = Mock()
    launch.report = {}
    launch.ensure.side_effect = startup.StartupError("模型失败")
    monkeypatch.setattr(startup, "Launcher", lambda: launch)
    monkeypatch.setattr(startup.sys, "argv", ["open_audioscope.py"])
    run = Mock()
    monkeypatch.setattr(startup.subprocess, "run", run)
    assert startup.main() == 1
    run.assert_not_called()
    assert launch.report["status"] == "failed"


def test_double_click_uses_single_startup_lock(root, monkeypatch):
    monkeypatch.setattr(startup, "ROOT", root)
    monkeypatch.setattr(startup.sys, "argv", ["open_audioscope.py"])
    constructor = Mock()
    monkeypatch.setattr(startup, "Launcher", constructor)
    folder = root / "runtime/startup"
    folder.mkdir(parents=True)
    with (folder / "launch.lock").open("a") as lock:
        startup.fcntl.flock(lock, startup.fcntl.LOCK_EX | startup.fcntl.LOCK_NB)
        assert startup.main() == 0
    constructor.assert_not_called()


def test_unhealthy_service_retries_only_once(root, monkeypatch):
    monkeypatch.setattr(startup, "port_busy", lambda: False)
    launch = FakeLauncher(root)
    launch.wait = Mock(side_effect=[True, False, False])
    with pytest.raises(startup.StartupError, match="停止自动重试"):
        launch.ensure()
    assert launch.calls.count(("restart", "demo")) == 1


def test_missing_volume_does_not_create_empty_history(root, monkeypatch):
    monkeypatch.setattr(startup, "port_busy", lambda: False)
    launch = FakeLauncher(root)
    launch.container = lambda service: None
    launch.command = Mock(side_effect=["image exists", None])
    with pytest.raises(startup.StartupError, match="数据卷不存在"):
        launch.ensure()
    assert not any(call[0] == "up" for call in launch.calls)


def test_health_validates_page_identity(monkeypatch):
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    opener = Mock()
    opener.open.return_value = response
    monkeypatch.setattr(workbench_health.urllib.request, "build_opener", lambda *a: opener)
    page = {
        "title": "AudioScope",
        "components": [{"props": {"elem_id": "scope-header"}}],
        "dependencies": [{"api_name": "analyse"}],
    }
    response.read.return_value = json.dumps(page)
    assert workbench_health.page_ready("http://127.0.0.1:7860")
    response.read.return_value = json.dumps({**page, "title": "Other app"})
    assert not workbench_health.page_ready("http://127.0.0.1:7860")
