"""Load user code referenced as 'module:function' or 'path/to/file.py:function'."""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import sys
import threading
from pathlib import Path
from typing import Callable

_LOCK = threading.RLock()  # two threads importing the same user file must not see a half-initialised module


def load_callable(spec: str, base_dir: str | Path | None = None) -> Callable:
    if ":" not in spec:
        raise ValueError(f"expected 'module:function', got {spec!r}")
    module_ref, func_name = spec.rsplit(":", 1)
    base = Path(base_dir or ".").resolve()

    if module_ref.lower().endswith(".py"):  # BOT.PY on Windows
        path = (base / module_ref).resolve()
        mod_name = f"_evalsprint_user_{path.stem}_{abs(hash(str(path)))}"
        with _LOCK:
            if mod_name in sys.modules:
                module = sys.modules[mod_name]
            else:
                # explicit loader: spec_from_file_location matches the suffix case-sensitively, so BOT.PY would find none
                spec_obj = importlib.util.spec_from_file_location(mod_name, path, loader=importlib.machinery.SourceFileLoader(mod_name, str(path)))
                if spec_obj is None or spec_obj.loader is None:
                    raise ImportError(f"cannot load {path}")
                module = importlib.util.module_from_spec(spec_obj)
                sys.modules[mod_name] = module
                if str(path.parent) not in sys.path:
                    sys.path.insert(0, str(path.parent))
                try:
                    spec_obj.loader.exec_module(module)
                except BaseException:
                    sys.modules.pop(mod_name, None)  # a module that failed to import must not be served half-built on the next call
                    raise
    else:
        with _LOCK:
            if str(base) not in sys.path:
                sys.path.insert(0, str(base))
            module = importlib.import_module(module_ref)

    try:
        return getattr(module, func_name)
    except AttributeError as exc:
        raise AttributeError(f"{module_ref} has no attribute {func_name!r}") from exc
