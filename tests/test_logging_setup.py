"""Tests for the rotating diagnostics log file (2026-09-08 incident fix).

Before this change the server wrote no log file at all — an IBKR rejection
reason logged via self.logger.* scrolled past in stderr and was gone. These
tests check the resolution rule (default path / IBKR_MCP_LOG_FILE override /
empty-string-disables) and that setup_logging installs a RotatingFileHandler
with the expected size/backup-count when a path is in effect.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler

import pytest

# ibkr_mcp_server/__init__.py does `from .main import main, cli`, which
# rebinds the package attribute `ibkr_mcp_server.main` to the async `main()`
# function — shadowing the submodule of the same name. Pull the real
# submodule out of sys.modules to dodge that rather than `import
# ibkr_mcp_server.main as main_mod` (which resolves via the clobbered
# attribute and hands back the function).
import ibkr_mcp_server.main  # noqa: F401 (ensures it's imported)
main_mod = sys.modules["ibkr_mcp_server.main"]


@pytest.fixture(autouse=True)
def _reset_root_logger():
    """setup_logging() uses logging.basicConfig(force=True), which mutates
    the root logger's handlers globally — restore them after each test so
    this file doesn't leak file handles into the rest of the suite."""
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    yield
    for h in root.handlers[:]:
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    for h in original_handlers:
        root.addHandler(h)
    root.setLevel(original_level)


def test_default_log_file_is_under_dot_trader_logs(monkeypatch):
    monkeypatch.delenv("IBKR_MCP_LOG_FILE", raising=False)
    assert main_mod._resolve_log_file() == main_mod.DEFAULT_LOG_FILE
    assert main_mod.DEFAULT_LOG_FILE.endswith("/.trader/logs/ibkr_mcp_server.log")


def test_env_var_overrides_default_path(monkeypatch, tmp_path):
    custom = str(tmp_path / "custom.log")
    monkeypatch.setenv("IBKR_MCP_LOG_FILE", custom)
    assert main_mod._resolve_log_file() == custom


def test_empty_env_var_disables_file_logging(monkeypatch):
    monkeypatch.setenv("IBKR_MCP_LOG_FILE", "")
    assert main_mod._resolve_log_file() is None


def test_empty_env_var_with_whitespace_disables_file_logging(monkeypatch):
    monkeypatch.setenv("IBKR_MCP_LOG_FILE", "   ")
    assert main_mod._resolve_log_file() is None


def test_explicit_argument_wins_over_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("IBKR_MCP_LOG_FILE", str(tmp_path / "from_env.log"))
    explicit = str(tmp_path / "from_flag.log")
    assert main_mod._resolve_log_file(explicit) == explicit


def test_setup_logging_installs_rotating_file_handler(tmp_path):
    log_file = tmp_path / "nested" / "ibkr_mcp_server.log"
    main_mod.setup_logging(level="INFO", log_file=str(log_file), mcp_mode=True)

    root = logging.getLogger()
    file_handlers = [h for h in root.handlers if isinstance(h, RotatingFileHandler)]
    assert len(file_handlers) == 1
    fh = file_handlers[0]
    assert fh.maxBytes == main_mod.LOG_MAX_BYTES
    assert fh.backupCount == main_mod.LOG_BACKUP_COUNT
    # Parent directory did not exist — setup_logging must create it.
    assert log_file.parent.is_dir()


def test_setup_logging_disabled_installs_no_file_handler(tmp_path):
    main_mod.setup_logging(level="INFO", log_file=None, mcp_mode=True)
    root = logging.getLogger()
    file_handlers = [h for h in root.handlers if isinstance(h, RotatingFileHandler)]
    assert file_handlers == []


def test_setup_logging_mcp_mode_never_writes_to_stdout(tmp_path, capsys):
    """stdio MCP transport: stdout must carry only protocol frames. In
    mcp_mode, logging must go to stderr and/or the file, never stdout."""
    log_file = tmp_path / "ibkr_mcp_server.log"
    main_mod.setup_logging(level="INFO", log_file=str(log_file), mcp_mode=True)
    logging.getLogger("ibkr_mcp_server.client").info("test order placement message")

    import sys as _sys
    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler):
            assert h.stream is not _sys.stdout

    captured = capsys.readouterr()
    assert captured.out == ""
