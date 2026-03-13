"""Slither helpers – install solc, run Slither, parse results."""
from __future__ import annotations
import json, os, subprocess, sys, shutil, tempfile
from pathlib import Path
from solc_select import solc_select  # type: ignore
from .config import REENTRANCY_DETECTORS, slither_executable, solc_executable

_IS_WIN = sys.platform == "win32"


def _run_slither_cmd(sol_path: Path, solc_version: str,
                      extra_args: list[str], timeout: int = 60) -> str:
    """Run Slither via os.system to avoid Python subprocess handle leaks.
    Returns the raw stdout/output as a string."""
    out_path = tempfile.mktemp(suffix=".slither_out")
    slither = slither_executable()
    solc = solc_executable()
    parts = [
        f'"{slither}"',
        f'"{sol_path}"',
        "--solc", f'"{solc}"',
        "--solc-solcs-select", solc_version,
    ] + extra_args
    cmd_line = " ".join(parts)
    try:
        os.system(f'{cmd_line} 2>NUL')
        if os.path.exists(out_path):
            with open(out_path, "r", errors="ignore") as f:
                return f.read()
        return ""
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


# ── Solc management ────────────────────────────────────────────────────────
def ensure_solc(version: str) -> None:
    """Install the given solc version via solc-select if not already present."""
    try:
        installed = solc_select.installed_versions()
    except Exception:
        installed = []
    if version not in installed:
        solc_select.install_artifacts([version])
    solc_select.switch_global_version(version, always_install=True)


# ── Slither JSON run ──────────────────────────────────────────────────────
def run_slither_json(sol_path: Path, solc_version: str, timeout: int = 60) -> dict:
    """Run Slither with --json <file> and return parsed JSON (or empty dict on failure)."""
    ensure_solc(solc_version)
    json_out = tempfile.mktemp(suffix=".json")
    slither = slither_executable()
    solc = solc_executable()
    dets = ",".join(REENTRANCY_DETECTORS)
    cmd = (
        f'"{slither}" "{sol_path}" '
        f'--solc "{solc}" --solc-solcs-select {solc_version} '
        f'--json "{json_out}" --detect {dets} 2>NUL'
    )
    try:
        os.system(cmd)
        if os.path.exists(json_out):
            with open(json_out, "r", errors="ignore") as f:
                text = f.read()
            return json.loads(text) if text.strip() else {}
        return {}
    except Exception:
        return {}
    finally:
        try:
            os.unlink(json_out)
        except OSError:
            pass


def slither_has_reentrancy(sol_path: Path, solc_version: str) -> bool:
    """Return True if Slither flags any reentrancy detector."""
    data = run_slither_json(sol_path, solc_version)
    for det in data.get("results", {}).get("detectors", []):
        if det.get("check", "") in REENTRANCY_DETECTORS:
            return True
    return False


def slither_reentrancy_count(sol_path: Path, solc_version: str) -> int:
    """Return the number of reentrancy findings Slither reports."""
    data = run_slither_json(sol_path, solc_version)
    count = 0
    for det in data.get("results", {}).get("detectors", []):
        if det.get("check", "") in REENTRANCY_DETECTORS:
            count += 1
    return count


# ── CFG extraction ────────────────────────────────────────────────────────
def extract_cfg_stats(sol_path: Path, solc_version: str, timeout: int = 60) -> dict:
    """
    Run `slither <file> --print cfg` and count nodes/edges from the .dot output.
    Returns {"cfg_nodes": int, "cfg_edges": int}.
    """
    ensure_solc(solc_version)
    output = _run_slither_cmd(sol_path, solc_version,
                               ["--print", "cfg"], timeout)
    nodes, edges = 0, 0
    for line in output.splitlines():
        line_s = line.strip()
        if line_s.startswith('"') and "->" in line_s:
            edges += 1
        elif line_s.startswith('"') and "[" in line_s:
            nodes += 1
    return {"cfg_nodes": nodes, "cfg_edges": edges}
