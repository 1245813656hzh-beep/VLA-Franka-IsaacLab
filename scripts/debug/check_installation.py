#!/usr/bin/env python3
"""
Installation verification script.

Run this to check if your environment is correctly configured for
VLA-Franka-IsaacLab.

Usage:
    python scripts/debug/check_installation.py
"""

from __future__ import annotations

import importlib
import sys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check(module: str, required: bool = True, min_version: str | None = None) -> bool:
    """Check if a module is importable and optionally meets version requirement."""
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, "__version__", "unknown")
        ok = True
        msg = f"✓ {module} ({version})"
        if min_version and version != "unknown":
            try:
                from packaging import version as pkg_version
                if pkg_version.parse(version) < pkg_version.parse(min_version):
                    ok = False
                    msg = f"✗ {module} ({version}) — requires >= {min_version}"
            except ImportError:
                # Fallback: simple tuple comparison for x.y.z versions
                def _parse(v):
                    return tuple(int(x) for x in v.split(".")[:3])
                if _parse(version) < _parse(min_version):
                    ok = False
                    msg = f"✗ {module} ({version}) — requires >= {min_version}"
    except ImportError:
        ok = False
        msg = f"✗ {module} — NOT INSTALLED"

    if required or ok:
        print(f"  {msg}")
    else:
        print(f"  ○ {msg} (optional)")
    return ok


def _check_env_var(name: str, hint: str = "") -> bool:
    import os
    value = os.environ.get(name, "")
    if value:
        print(f"  ✓ {name}={value}")
        return True
    else:
        print(f"  ○ {name} not set{f' ({hint})' if hint else ''}")
        return False


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_base() -> bool:
    print("\n" + "=" * 60)
    print("Base Dependencies")
    print("=" * 60)
    ok = True
    ok &= _check("numpy", min_version="1.24.0")
    ok &= _check("torch", min_version="2.0.0")
    ok &= _check("gymnasium", min_version="0.28.0")
    ok &= _check("cv2", required=False)
    ok &= _check("msgpack", required=False)
    ok &= _check("zmq", required=False)
    return ok


def check_isaaclab() -> bool:
    print("\n" + "=" * 60)
    print("IsaacLab Environment")
    print("=" * 60)
    ok = True
    ok &= _check("isaaclab", min_version="0.48.0")
    ok &= _check("isaaclab_tasks", required=False)
    ok &= _check("isaaclab_assets", required=False)
    ok &= _check("isaaclab_mimic", required=False)
    _check_env_var("ISAACSIM_PATH", "Isaac Sim installation directory")
    return ok


def check_lerobot() -> bool:
    print("\n" + "=" * 60)
    print("LeRobot")
    print("=" * 60)
    ok = _check("lerobot", min_version="0.4.0")
    if ok:
        import lerobot
        print(f"  → LeRobot path: {lerobot.__file__}")
    return ok


def check_transformers() -> bool:
    print("\n" + "=" * 60)
    print("Transformers (for ACT inference)")
    print("=" * 60)
    ok = _check("transformers", required=False, min_version="4.40.0")
    if ok:
        import transformers
        has_sliding = hasattr(transformers.cache_utils, "SlidingWindowCache")
        print(f"  → SlidingWindowCache: {'available' if has_sliding else 'MISSING (may need newer version)'}")
    return ok


def check_gr00t() -> bool:
    print("\n" + "=" * 60)
    print("GR00T-N1.5")
    print("=" * 60)
    ok = _check("gr00t", required=False)
    _check_env_var("GR00T_PATH", "path to GR00T-N1.5 source directory")
    return ok


def check_project_structure() -> bool:
    print("\n" + "=" * 60)
    print("Project Structure")
    print("=" * 60)
    import os
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    checks = [
        ("tasks/franka", "Task configurations"),
        ("scripts/data_collection", "Data collection scripts"),
        ("scripts/inference", "Inference scripts"),
        ("configs", "Config examples"),
    ]
    ok = True
    for subdir, desc in checks:
        path = project_root / subdir
        exists = path.exists()
        print(f"  {'✓' if exists else '✗'} {subdir}/ — {desc}")
        ok &= exists
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("VLA-Franka-IsaacLab Installation Check")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")

    results = {
        "base": check_base(),
        "isaaclab": check_isaaclab(),
        "lerobot": check_lerobot(),
        "transformers": check_transformers(),
        "gr00t": check_gr00t(),
        "structure": check_project_structure(),
    }

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    all_ok = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        symbol = "✓" if ok else "✗"
        print(f"  {symbol} {name:15s} {status}")
        all_ok &= ok

    print("\n" + ("All checks passed!" if all_ok else "Some checks failed. See details above."))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
