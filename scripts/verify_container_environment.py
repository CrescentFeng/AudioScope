"""Fresh PostgreSQL migration and installed-source/dependency evidence.

Creates and removes only its own randomly named empty acceptance database.
Never drops, resets or truncates the application database or its volume.
"""

import json
import os
import subprocess
import uuid
from pathlib import Path

from sqlalchemy import create_engine, inspect

from audioscope.release_catalog import sessions
from audioscope.software_validation import write_json
from audioscope.workbench import sha256


def command(args, env=None):
    result = subprocess.run(args, env=env, text=True, capture_output=True)
    if result.returncode:
        # Do not include command environment (which contains the database password).
        raise RuntimeError(f"Container verification command failed (exit {result.returncode})")
    return result.stdout


def run(root):
    factory = sessions(root)
    database_url = factory.kw["bind"].url
    name = "audioscope_acceptance_" + uuid.uuid4().hex
    admin = create_engine(database_url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    isolated = create_engine(database_url.set(database=name))
    created = False
    report = {"status": "running", "physical_validation_performed": False}
    try:
        with admin.connect() as connection:
            connection.exec_driver_sql(f'CREATE DATABASE "{name}"')
        created = True
        env = dict(os.environ)
        env["AUDIOSCOPE_DATABASE_URL"] = database_url.set(
            host="db", port=5432, database=name
        ).render_as_string(hide_password=False)
        migration = [
            "docker",
            "compose",
            "run",
            "--rm",
            "--no-deps",
            "-e",
            "AUDIOSCOPE_DATABASE_URL",
            "migrate",
            "alembic",
            "upgrade",
            "head",
        ]
        command(migration, env)
        inspector = inspect(isolated)
        catalog_tables = sorted(inspector.get_table_names(schema="catalog"))
        public_tables = sorted(inspector.get_table_names(schema="public"))
        if not {"workbench_jobs", "workbench_reviews", "alembic_version"} <= set(public_tables):
            raise RuntimeError("Fresh workbench migration incomplete")
        if not {
            "audio_files",
            "processing_runs",
            "model_artifacts",
            "benchmark_results",
            "release_audits",
        } <= set(catalog_tables):
            raise RuntimeError("Fresh catalog migration incomplete")
        command(migration, env)
        with isolated.connect() as connection:
            revision = connection.exec_driver_sql(
                "SELECT version_num FROM public.alembic_version"
            ).scalar_one()
        if revision != "0002_workbench":
            raise RuntimeError("Unexpected migration head")
        report["fresh_migration"] = {
            "revision": revision,
            "catalog_tables": catalog_tables,
            "public_tables": public_tables,
            "second_upgrade_succeeded": True,
        }
        probe = """import audioscope, hashlib, json, pathlib, importlib.metadata as m
p = pathlib.Path(audioscope.__file__).parent
source = {str(f.relative_to(p)): hashlib.sha256(f.read_bytes()).hexdigest()
          for f in p.rglob('*.py')}
deps = [{'name': d.metadata['Name'], 'version': d.version} for d in m.distributions()]
print(json.dumps({'source': source,
                  'dependencies': sorted(deps, key=lambda d:d['name'].lower())}))"""
        runtime = json.loads(
            command(["docker", "compose", "exec", "-T", "api", "python", "-c", probe])
        )
        local = {
            str(path.relative_to(root / "src/audioscope")): sha256(path)
            for path in (root / "src/audioscope").rglob("*.py")
        }
        if runtime["source"] != local:
            raise RuntimeError("Installed container source differs from current repository")
        report["runtime"] = runtime
        report["installed_source_matches"] = True
        report["pip_check"] = command(
            ["docker", "compose", "exec", "-T", "api", "python", "-m", "pip", "check"]
        ).strip()
        report["image_id"] = command(
            ["docker", "image", "inspect", "audioscope-python:local", "--format", "{{.Id}}"]
        ).strip()
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        isolated.dispose()
        if created:
            with admin.connect() as connection:
                connection.exec_driver_sql(f'DROP DATABASE "{name}"')
            report["temporary_empty_database_removed"] = True
        admin.dispose()
        write_json(root / "reports/generated/workbench/container-environment.json", report)
    print("Fresh migration, repeat migration, installed-source hashes and dependencies passed.")


if __name__ == "__main__":
    run(Path.cwd())
