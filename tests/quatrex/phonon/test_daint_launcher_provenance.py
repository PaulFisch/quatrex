import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "daint_launcher", ROOT / "phonon/scripts/daint.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_source_commit_records_remote_checkout(monkeypatch, capsys):
    monkeypatch.setattr(
        MODULE.subprocess, "check_output", lambda *args, **kwargs: "local\n"
    )
    monkeypatch.setattr(MODULE, "ssh", lambda *args, **kwargs: "remote\n")

    exports = MODULE._source_commit_env()

    assert exports.splitlines() == [
        "export QX_SOURCE_COMMIT=remote",
        "export QX_LOCAL_SOURCE_COMMIT=local",
    ]
    assert "local and Daint commits differ" in capsys.readouterr().err


def test_source_commit_omits_duplicate_local_audit(monkeypatch, capsys):
    monkeypatch.setattr(
        MODULE.subprocess, "check_output", lambda *args, **kwargs: "same\n"
    )
    monkeypatch.setattr(MODULE, "ssh", lambda *args, **kwargs: "same\n")

    assert MODULE._source_commit_env() == "export QX_SOURCE_COMMIT=same"
    assert capsys.readouterr().err == ""


def test_pull_excludes_restart_checkpoints(monkeypatch):
    calls = []
    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda args, check: calls.append((args, check)),
    )

    MODULE.cmd_pull(SimpleNamespace(name="sample"))

    args, check = calls.pop()
    assert "--exclude=sigma*.npz" in args
    assert "--exclude=sigma*.npz.shards/" in args
    assert args[-1] == "cluster/sample/"
    assert check
