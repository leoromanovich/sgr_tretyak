from __future__ import annotations

import atexit
from datetime import datetime
from pathlib import Path
import os
import sys
from typing import Optional

from ..config import settings


class _StreamTee:
    def __init__(self, primary, secondary):
        self.primary = primary
        self.secondary = secondary

    def write(self, data: str) -> int:
        self.primary.write(data)
        self.secondary.write(data)
        return len(data)

    def flush(self) -> None:
        self.primary.flush()
        self.secondary.flush()


_log_path: Optional[Path] = None
_log_file = None
_stdout_original = sys.stdout
_stderr_original = sys.stderr


def _shutdown_logging() -> None:
    global _log_file, _log_path
    if _log_file is None:
        return
    sys.stdout = _stdout_original
    sys.stderr = _stderr_original
    _log_file.flush()
    _log_file.close()
    _log_file = None
    _log_path = None


def setup_file_logging(prefix: str = "run") -> Path:
    """
    Перенаправляет stdout/stderr так, чтобы все выводы дублировались в logs/<prefix>_<ts>.log.
    Повторные вызовы возвращают уже созданный путь.
    """
    global _log_path, _log_file
    if _log_path is not None:
        return _log_path

    logs_dir = settings.project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_path = logs_dir / f"{prefix}_{timestamp}.log"
    _log_file = open(_log_path, "w", encoding="utf-8")

    sys.stdout = _StreamTee(_stdout_original, _log_file)
    sys.stderr = _StreamTee(_stderr_original, _log_file)

    atexit.register(_shutdown_logging)
    _stdout_original.write(f"[log] Вывод дублируется в {_log_path}\n")
    _stdout_original.flush()
    return _log_path
