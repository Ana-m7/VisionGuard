"""Everything needed to say exactly WHAT was measured (audit issues #4, #16):
git commit + dirty state, a snapshot of every config value, package
versions, the tracker in use, the machine, and the run arguments.

Only read-only git commands are used (rev-parse, status --porcelain).
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime

from . import paths

import config  # noqa: E402


def _git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=paths.PROJECT_ROOT, capture_output=True,
                             text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def git_info() -> dict:
    porcelain = _git("status", "--porcelain") or ""
    lines = [ln for ln in porcelain.splitlines() if ln.strip()]
    suite_rel = os.path.relpath(paths.SUITE_DIR, paths.PROJECT_ROOT).replace("\\", "/")
    outside = [ln for ln in lines if suite_rel not in ln.replace("\\", "/")]
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(lines),
        "dirty_files_outside_suite": outside,
        "note": "dirty_files_outside_suite must be empty for results to describe the commit",
    }


def config_snapshot() -> dict:
    snap = {}
    for k in sorted(dir(config)):
        if not k.isupper():
            continue
        v = getattr(config, k)
        if any(s in k for s in ("TOKEN", "SID", "SECRET", "PASSWORD")):
            v = "<redacted>" if v else ""
        try:
            json.dumps(v)
            snap[k] = v
        except TypeError:
            snap[k] = repr(v)
    return snap


def package_versions() -> dict:
    vers = {"python": sys.version.split()[0], "python_executable": sys.executable}
    for mod in ("ultralytics", "torch", "torchvision", "cv2", "numpy", "scipy",
                "matplotlib", "requests"):
        try:
            m = __import__(mod)
            vers[mod] = getattr(m, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            vers[mod] = None
    return vers


def root_requirements_pins() -> dict:
    pins = {}
    path = os.path.join(paths.PROJECT_ROOT, "requirements.txt")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if "==" in line:
                    k, v = line.strip().split("==", 1)
                    pins[k.lower()] = v
    return pins


def machine() -> dict:
    cpu = platform.processor()
    try:
        if sys.platform == "win32":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        device = "unknown"
    return {"cpu": cpu, "os": platform.platform(), "inference_device": device,
            "logical_cpus": os.cpu_count()}


def collect(args: dict, tracker: dict, parity: dict) -> dict:
    vers = package_versions()
    pins = root_requirements_pins()
    mismatches = {}
    for pkg, mod in (("ultralytics", "ultralytics"), ("torch", "torch"),
                     ("numpy", "numpy"), ("opencv-python", "cv2"), ("torchvision", "torchvision")):
        if pkg in pins and vers.get(mod) and not str(vers[mod]).startswith(pins[pkg]):
            mismatches[pkg] = {"installed": vers[mod], "root_requirements_pin": pins[pkg]}
    return {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "git": git_info(),
        "config_snapshot": config_snapshot(),
        "versions": vers,
        "root_requirements_mismatches": mismatches,
        "tracker": tracker,
        "production_parity": parity,
        "machine": machine(),
        "args": args,
    }
