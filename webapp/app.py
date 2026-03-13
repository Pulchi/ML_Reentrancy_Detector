"""
Flask web demo: upload a .sol file → run Slither → extract features → predict reentrancy
→ explain the result with feature-based reasoning (+ optional Groq LLM).
"""

import os, re, tempfile, json, shutil, subprocess, glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

# Load .env file if present (so user can put GROQ_API_KEY there)
_env_file = Path(__file__).resolve().parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024  # 512 KB max upload

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"

# Load model artifacts
model = joblib.load(MODEL_DIR / "rf_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
feature_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")

# Check if Slither is available (check venv first, then PATH)
SLITHER_BIN = None
_venv_slither = Path(__file__).resolve().parent.parent.parent / ".venv" / "Scripts" / "slither.exe"
if _venv_slither.exists():
    SLITHER_BIN = str(_venv_slither)
elif shutil.which("slither"):
    SLITHER_BIN = shutil.which("slither")

if SLITHER_BIN:
    print(f"[INFO] Slither found at {SLITHER_BIN} — will use for accurate feature extraction")
else:
    print("[WARN] Slither not found — falling back to regex-only feature extraction")

REENTRANCY_CHECKS = {
    "reentrancy-eth", "reentrancy-no-eth", "reentrancy-benign",
    "reentrancy-unlimited-gas", "reentrancy-events",
}


def _extract_function_bodies(code):
    """Extract function bodies using brace counting (handles nested blocks)."""
    bodies = []
    for m in re.finditer(r'\bfunction\s+\w+[^{]*\{', code):
        start = m.end()
        depth = 1
        i = start
        while i < len(code) and depth > 0:
            if code[i] == '{':
                depth += 1
            elif code[i] == '}':
                depth -= 1
            i += 1
        bodies.append(code[start:i - 1])
    return bodies


# ── Slither-based feature extraction ───────────────────────────────
# Map pragma versions to installed solc versions
SOLC_SELECT_BIN = None
SOLC_BIN = None
_venv_solc_select = Path(__file__).resolve().parent.parent.parent / ".venv" / "Scripts" / "solc-select.exe"
_venv_solc = Path(__file__).resolve().parent.parent.parent / ".venv" / "Scripts" / "solc.exe"
if _venv_solc_select.exists():
    SOLC_SELECT_BIN = str(_venv_solc_select)
if _venv_solc.exists():
    SOLC_BIN = str(_venv_solc)
elif shutil.which("solc"):
    SOLC_BIN = shutil.which("solc")


def _detect_solc_version(code):
    """Detect required solc version from pragma and return best installed match."""
    m = re.search(r'pragma\s+solidity\s+[\^~>=<]*\s*(0\.(\d+)\.\d+)', code)
    if not m:
        return None
    minor = int(m.group(2))
    # Map to our installed versions: 0.4.24, 0.5.0, 0.8.0
    version_map = {4: "0.4.24", 5: "0.5.0", 6: "0.6.12", 7: "0.7.6", 8: "0.8.0"}
    return version_map.get(minor)


def _switch_solc(version):
    """Switch solc to the specified version using solc-select."""
    if not SOLC_SELECT_BIN or not version:
        return False
    try:
        # First try to use it; if not installed, install it
        result = subprocess.run([SOLC_SELECT_BIN, "use", version],
                                capture_output=True, timeout=30, text=True)
        if result.returncode != 0:
            # Try installing the version first
            print(f"[INFO] Installing solc {version}...")
            subprocess.run([SOLC_SELECT_BIN, "install", version],
                           capture_output=True, timeout=120, text=True)
            subprocess.run([SOLC_SELECT_BIN, "use", version],
                           capture_output=True, timeout=30, text=True)
        print(f"[INFO] Switched solc to {version}")
        return True
    except Exception as e:
        print(f"[WARN] Failed to switch solc: {e}")
        return False


def _create_import_stubs(code, work_dir):
    """Create empty stub files for imports so Slither can compile.
    E.g. import '@openzeppelin/contracts/math/SafeMath.sol' creates that path."""
    imports = re.findall(r'''import\s+['"]([^'"]+)['"]''', code)
    imports += re.findall(r'''import\s+\{[^}]*\}\s+from\s+['"]([^'"]+)['"]''', code)
    for imp in imports:
        # Resolve relative to work_dir
        imp_path = Path(work_dir) / imp.lstrip("./")
        imp_path.parent.mkdir(parents=True, exist_ok=True)
        if not imp_path.exists():
            # Create a minimal stub with the same pragma
            pragma = re.search(r'pragma\s+solidity\s+[^;]+;', code)
            pragma_line = pragma.group() if pragma else "pragma solidity ^0.8.0;"
            # Extract what's being imported (library/contract names)
            stub = f"// SPDX-License-Identifier: MIT\n{pragma_line}\n"
            # Check if SafeMath or similar library is imported
            if "SafeMath" in imp:
                stub += "library SafeMath {\n"
                stub += "    function add(uint256 a, uint256 b) internal pure returns (uint256) { return a + b; }\n"
                stub += "    function sub(uint256 a, uint256 b) internal pure returns (uint256) { return a - b; }\n"
                stub += "    function mul(uint256 a, uint256 b) internal pure returns (uint256) { return a * b; }\n"
                stub += "    function div(uint256 a, uint256 b) internal pure returns (uint256) { return a / b; }\n"
                stub += "}\n"
            elif "ReentrancyGuard" in imp:
                stub += "contract ReentrancyGuard {\n"
                stub += "    bool private _notEntered;\n"
                stub += "    modifier nonReentrant() { require(_notEntered); _notEntered = false; _; _notEntered = true; }\n"
                stub += "}\n"
            else:
                # Generic empty contract stub
                name = Path(imp).stem
                stub += f"contract {name} {{}}\n"
            imp_path.write_text(stub, encoding="utf-8")
            print(f"[INFO] Created stub for import: {imp}")


def run_slither(sol_path, work_dir, code):
    """Run Slither on a .sol file, return (json_data, dot_files)."""
    # Auto-detect and switch solc version
    needed_version = _detect_solc_version(code)
    if needed_version:
        _switch_solc(needed_version)

    # Create stub files for imports so Slither can compile
    _create_import_stubs(code, work_dir)

    json_path = Path(work_dir) / "slither_output.json"

    # Run Slither JSON analysis
    cmd_json = [
        SLITHER_BIN, str(sol_path),
        "--json", str(json_path),
        "--no-fail",
    ]
    if SOLC_BIN:
        cmd_json += ["--solc", SOLC_BIN]
    try:
        result = subprocess.run(cmd_json, capture_output=True, timeout=60,
                                cwd=work_dir, text=True)
        print(f"[DEBUG] Slither JSON exit code: {result.returncode}")
        if result.stderr:
            # Only print first 500 chars of stderr
            print(f"[DEBUG] Slither stderr: {result.stderr[:500]}")
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"[WARN] Slither JSON failed: {e}")
        return None, []

    json_data = None
    if json_path.exists():
        try:
            json_data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    # Run Slither CFG printer for DOT files
    cmd_dot = [
        SLITHER_BIN, str(sol_path),
        "--print", "cfg",
        "--no-fail",
    ]
    if SOLC_BIN:
        cmd_dot += ["--solc", SOLC_BIN]
    try:
        result = subprocess.run(cmd_dot, capture_output=True, timeout=60,
                                cwd=work_dir, text=True)
        print(f"[DEBUG] Slither CFG exit code: {result.returncode}")
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"[WARN] Slither CFG failed: {e}")

    dot_files = glob.glob(str(Path(work_dir) / "*.dot"))
    dot_files = [f for f in dot_files if "call-graph" not in f]
    print(f"[DEBUG] DOT files found in {work_dir}: {[Path(f).name for f in dot_files]}")

    return json_data, dot_files


def json_features(json_data):
    """Extract has_reentrancy_eth and num_reentrancy_detectors from Slither JSON."""
    has_eth = 0
    num_det = 0
    if json_data:
        for d in json_data.get("results", {}).get("detectors", []):
            check = d.get("check", "")
            if check in REENTRANCY_CHECKS:
                num_det += 1
                if check == "reentrancy-eth":
                    has_eth = 1
    return {"has_reentrancy_eth": has_eth, "num_reentrancy_detectors": num_det}


def parse_dot(dot_path):
    """Parse a DOT file into nodes and edges."""
    text = Path(dot_path).read_text(encoding="utf-8", errors="replace")
    nodes = {}
    edges = []
    for m in re.finditer(r'(\d+)\[label="(.*?)"\];', text, re.DOTALL):
        nodes[int(m.group(1))] = m.group(2)
    for m in re.finditer(r'(\d+)->(\d+)(?:\[.*?\])?;', text):
        edges.append((int(m.group(1)), int(m.group(2))))
    return nodes, edges


def _has_ext_call(label):
    u = label.upper()
    if "LOW_LEVEL_CALL" in u or "HIGH_LEVEL_CALL" in u:
        return True
    if re.search(r'\.(CALL|SEND|TRANSFER)\s*[\.(]', label, re.IGNORECASE):
        return True
    return False


def _has_state_write(label):
    return bool(re.search(r'\(->\s*[a-zA-Z]', label))


def cfg_features(dot_files):
    """Extract CEI violation and external call count from DOT CFG files."""
    cei = 0
    total_ext = 0
    print(f"[DEBUG] cfg_features called with {len(dot_files)} DOT files")
    for df in dot_files:
        nodes, edges = parse_dot(df)
        if not nodes:
            continue
        adj = defaultdict(list)
        for s, d in edges:
            adj[s].append(d)
        entry = min(nodes.keys())
        visited = {entry}
        queue = [entry]
        seen_call = False
        while queue:
            n = queue.pop(0)
            lbl = nodes.get(n, "")
            if _has_ext_call(lbl):
                seen_call = True
            if seen_call and _has_state_write(lbl):
                cei = 1
            for nb in adj.get(n, []):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        total_ext += sum(1 for lbl in nodes.values() if _has_ext_call(lbl))

    return {"cei_violation": cei, "num_external_calls": total_ext}


def source_features(code):
    """Extract source-code features (13 features for v3 model)."""
    has_low_level = 1 if re.search(r'\.call', code) else 0
    uses_call = bool(re.search(r'\.call[\.\({\s]', code))
    has_dangerous = 1 if uses_call else 0
    num_funcs = len(re.findall(r'\bfunction\b', code))
    has_mod = 1 if re.search(r'\bmodifier\b', code) else 0
    has_guard = 1 if re.search(r'ReentrancyGuard|nonReentrant|noReentrant', code, re.IGNORECASE) else 0
    has_emit = 1 if re.search(r'\bemit\b', code) else 0

    has_bool_private = bool(re.search(r'\bbool\b.*\b\w+\b', code))
    has_require_not = bool(re.search(r'require\s*\(\s*!\s*\w+', code))
    has_mutex_pattern = 1 if (has_bool_private and has_require_not and not has_guard) else 0

    num_require_assert = len(re.findall(r'\b(?:require|assert)\s*\(', code))

    has_state_set_before_call = 0
    func_bodies = _extract_function_bodies(code)
    for body in func_bodies:
        lines = body.split('\n')
        found_assignment = False
        for line in lines:
            s = line.strip()
            if re.search(r'\b\w+\s*\[.*\]\s*[-+*/]?=', s) or re.search(r'\b\w+\s*[-+*/]?=\s*(?!.*\bfunction\b)', s):
                if not re.search(r'\b(uint|int|bool|address|string|bytes|mapping)\b', s):
                    found_assignment = True
            if found_assignment and re.search(r'\.(call|send|transfer)\s*[\.\({]', s, re.IGNORECASE):
                has_state_set_before_call = 1
                break

    modifier_defs = set(re.findall(r'modifier\s+(\w+)', code))
    num_modifiers_applied = 0
    for md in modifier_defs:
        num_modifiers_applied += len(re.findall(r'\b' + re.escape(md) + r'\b', code)) - 1

    has_internal_ext_call = 0
    internal_funcs = []
    for m in re.finditer(r'function\s+\w+[^{]*\b(?:internal|private)\b[^{]*\{', code):
        start = m.end()
        depth, i = 1, start
        while i < len(code) and depth > 0:
            if code[i] == '{': depth += 1
            elif code[i] == '}': depth -= 1
            i += 1
        internal_funcs.append(code[start:i-1])
    for body in internal_funcs:
        if re.search(r'\.(call|send|transfer)\s*[\.\({]', body, re.IGNORECASE):
            has_internal_ext_call = 1
            break
    if not has_internal_ext_call:
        internal_func_names = set(re.findall(r'function\s+(\w+)[^{]*\b(?:internal|private)\b', code))
        public_bodies = []
        for m in re.finditer(r'function\s+\w+[^{]*(?:public|external)[^{]*\{', code):
            start = m.end()
            depth, i = 1, start
            while i < len(code) and depth > 0:
                if code[i] == '{': depth += 1
                elif code[i] == '}': depth -= 1
                i += 1
            public_bodies.append(code[start:i-1])
        for body in public_bodies:
            for fn in internal_func_names:
                if re.search(r'\b' + re.escape(fn) + r'\s*\(', body):
                    has_internal_ext_call = 1
                    break

    has_interface_cast = 0
    interfaces = re.findall(r'\binterface\s+(\w+)', code)
    for iface in interfaces:
        if re.search(re.escape(iface) + r'\s*\([^)]*\)\s*\.', code):
            has_interface_cast = 1
            break
    if not has_interface_cast:
        if re.search(r'\b[A-Z]\w*\s*\(\s*\w+\s*\)\s*\.\s*\w+\s*\(', code):
            has_interface_cast = 1

    lines_of_code = len([l for l in code.split('\n') if l.strip()])

    return {
        "has_low_level_call": has_low_level,
        "has_dangerous_call": has_dangerous,
        "num_functions": num_funcs,
        "has_modifier": has_mod,
        "has_reentrancy_guard": has_guard,
        "has_emit_after_call": has_emit,
        "has_mutex_pattern": has_mutex_pattern,
        "num_require_assert": num_require_assert,
        "has_state_set_before_call": has_state_set_before_call,
        "num_modifiers_applied": num_modifiers_applied,
        "has_internal_ext_call": has_internal_ext_call,
        "has_interface_cast": has_interface_cast,
        "lines_of_code": lines_of_code,
    }


def extract_features_with_slither(code: str) -> tuple:
    """Run Slither + source analysis. Returns (features_dict, used_slither_bool, slither_raw)."""
    # Source features always work
    src_feats = source_features(code)
    slither_raw = {"detectors": [], "ran": False}

    if not SLITHER_BIN:
        # Fallback: regex-approximate Slither features
        src_feats.update(_regex_slither_features(code))
        return src_feats, False, slither_raw

    # Write code to temp file and run Slither
    tmp_dir = tempfile.mkdtemp(prefix="slither_")
    sol_path = Path(tmp_dir) / "contract.sol"
    try:
        sol_path.write_text(code, encoding="utf-8")
        json_data, dot_files = run_slither(sol_path, tmp_dir, code)

        if json_data is not None:
            # Extract raw Slither detectors for display
            slither_raw["ran"] = True
            for d in json_data.get("results", {}).get("detectors", []):
                slither_raw["detectors"].append({
                    "check": d.get("check", ""),
                    "impact": d.get("impact", ""),
                    "confidence": d.get("confidence", ""),
                    "description": d.get("description", ""),
                })

            j_feats = json_features(json_data)
            c_feats = cfg_features(dot_files) if dot_files else _regex_cfg_features(code)
            src_feats.update(j_feats)
            src_feats.update(c_feats)
            print(f"[INFO] Slither analysis complete: has_reentrancy_eth={j_feats['has_reentrancy_eth']}, "
                  f"num_det={j_feats['num_reentrancy_detectors']}, cei={c_feats.get('cei_violation', '?')}, "
                  f"ext_calls={c_feats.get('num_external_calls', '?')}")
            return src_feats, True, slither_raw
        else:
            print("[WARN] Slither returned no JSON, falling back to regex")
            src_feats.update(_regex_slither_features(code))
            return src_feats, False, slither_raw
    except Exception as e:
        print(f"[WARN] Slither error: {e}, falling back to regex")
        src_feats.update(_regex_slither_features(code))
        return src_feats, False, slither_raw
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _regex_slither_features(code):
    """Approximate all 4 Slither-dependent features from source regex."""
    j = _regex_json_features(code)
    c = _regex_cfg_features(code)
    j.update(c)
    return j


def _regex_json_features(code):
    """Approximate has_reentrancy_eth and num_reentrancy_detectors from source."""
    has_ext_call = bool(re.search(r'\.(call|send|transfer)\s*[\.\({]', code, re.IGNORECASE))
    has_state_after = False
    func_bodies = _extract_function_bodies(code)
    for body in func_bodies:
        lines = body.split('\n')
        seen_call = False
        for line in lines:
            s = line.strip()
            if re.search(r'\.(call|send|transfer)\s*[\.\({]', s, re.IGNORECASE):
                seen_call = True
            if seen_call and (re.search(r'\b\w+\s*\[.*\]\s*[-+*/]?=', s) or
                              re.search(r'\b\w+\s*[-+*/]?=\s*0\s*;', s)):
                has_state_after = True
                break

    has_reentrancy_eth = 1 if (has_ext_call and has_state_after) else 0
    num_reentrancy_detectors = 0
    if has_ext_call and has_state_after:
        num_reentrancy_detectors += 1
    if re.search(r'\.call\.value\(', code) or re.search(r'\.call\{value:', code):
        num_reentrancy_detectors += 1
    return {"has_reentrancy_eth": has_reentrancy_eth, "num_reentrancy_detectors": num_reentrancy_detectors}


def _regex_cfg_features(code):
    """Approximate CEI violation and external call count from source regex."""
    cei_violation = 0
    func_bodies = _extract_function_bodies(code)
    for body in func_bodies:
        lines = body.split('\n')
        seen_call = False
        for line in lines:
            s = line.strip()
            if re.search(r'\.(call|send|transfer)\s*[\.\({]', s, re.IGNORECASE):
                seen_call = True
            if seen_call and re.search(r'\b\w+\s*\[.*\]\s*[-+*/]?=', s):
                cei_violation = 1
                break
    num_external_calls = len(re.findall(r'\.(call|send|transfer)\s*[\.\({]', code, re.IGNORECASE))
    return {"cei_violation": cei_violation, "num_external_calls": num_external_calls}


# ── Explanation generator ───────────────────────────────────────────
def generate_explanations(features: dict, prediction: int, probability: float, code: str) -> tuple:
    """Generate rule-based and LLM explanations. Returns (rule_explanation, llm_explanation)."""
    rule_exp = _rule_based_explanation(features, prediction, probability)
    llm_exp = None

    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        try:
            llm_exp = _llm_explanation(features, prediction, probability, code, api_key)
            print("[INFO] Groq LLM explanation generated successfully")
        except Exception as e:
            print(f"[WARN] Groq LLM failed: {e}")
            llm_exp = f"[LLM Error] Groq API call failed: {e}. Showing rule-based analysis only."
    else:
        llm_exp = ("[LLM Unavailable] Set the GROQ_API_KEY environment variable to enable "
                   "AI-powered analysis using LLaMA 3.3 70B.\n\n"
                   "To enable:\n"
                   "  1. Get a free API key at https://console.groq.com\n"
                   "  2. Set it: set GROQ_API_KEY=your_key_here  (Windows)\n"
                   "     or: export GROQ_API_KEY=your_key_here  (Linux/Mac)\n"
                   "  3. Restart the app")
        print("[INFO] No GROQ_API_KEY set, LLM analysis unavailable")

    return rule_exp, llm_exp


def _llm_explanation(features, prediction, probability, code, api_key):
    from groq import Groq
    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": (
                "You are a smart contract security auditor specializing in reentrancy vulnerabilities. "
                "Analyze the given Solidity source code independently. "
                "Determine if it is vulnerable to reentrancy attacks. "
                "Identify specific vulnerable functions, lines, and patterns. "
                "Explain your reasoning step by step. Use bullet points. "
                "If vulnerable, explain the attack vector and recommend fixes. "
                "If safe, explain what protections are in place. "
                "Keep it under 400 words."
            )},
            {"role": "user", "content": (
                f"Analyze this Solidity contract for reentrancy vulnerabilities:\n\n"
                f"```solidity\n{code[:4000]}\n```\n\n"
                "Is this contract vulnerable to reentrancy? Explain your analysis."
            )},
        ],
        max_tokens=600,
        temperature=0.3,
    )
    return response.choices[0].message.content


def _rule_based_explanation(features, prediction, probability):
    """Generate explanation from feature values when no LLM is available."""
    lines = []
    label = "Reentrant (Vulnerable)" if prediction == 1 else "Safe"
    lines.append(f"**Prediction: {label}** (confidence: {probability:.1%})\n")

    if prediction == 1:  # Vulnerable
        lines.append("**Risk factors detected:**\n")
        if features["cei_violation"]:
            lines.append("- **CEI Violation**: External call is made BEFORE state variables are updated. "
                         "This is the classic reentrancy pattern — an attacker can re-enter the function "
                         "before the state is modified.")
        if features["has_dangerous_call"]:
            lines.append("- **Dangerous `.call()` usage**: The contract uses low-level `.call()` which "
                         "forwards all gas, allowing the callee to execute arbitrary code including callbacks.")
        if features["num_external_calls"] > 0:
            lines.append(f"- **External calls**: {features['num_external_calls']} external call(s) found, "
                         "each is a potential reentrancy entry point.")
        if features["has_internal_ext_call"]:
            lines.append("- **Hidden external call**: An external call is wrapped inside an internal/private "
                         "function, making the vulnerability harder to spot.")
        if features["has_interface_cast"]:
            lines.append("- **Interface casting**: The contract casts an address to an interface and calls it. "
                         "The target implementation is untrusted and could include callback logic.")
        if not features["has_reentrancy_guard"] and not features["has_mutex_pattern"]:
            lines.append("- **No reentrancy guard**: Neither `ReentrancyGuard`/`nonReentrant` modifier "
                         "nor a manual mutex pattern was detected.")

        lines.append("\n**Recommendations:**\n")
        lines.append("1. Apply the Checks-Effects-Interactions (CEI) pattern: update state BEFORE external calls.")
        lines.append("2. Use OpenZeppelin's `ReentrancyGuard` with `nonReentrant` modifier.")
        lines.append("3. Avoid low-level `.call()` when possible; use `.transfer()` or `.send()` for simple ETH transfers.")

    else:  # Safe
        lines.append("**Safety indicators detected:**\n")
        if features["has_reentrancy_guard"]:
            lines.append("- **ReentrancyGuard present**: The contract uses `nonReentrant` modifier, "
                         "which prevents reentrant calls.")
        if features["has_mutex_pattern"]:
            lines.append("- **Mutex pattern**: A boolean-based lock mechanism (require + flag toggle) "
                         "is detected, preventing reentrant execution.")
        if features["has_state_set_before_call"]:
            lines.append("- **State updated before call**: State variables are modified before the external call, "
                         "following the CEI pattern.")
        if features["has_emit_after_call"]:
            lines.append("- **Event emission**: Events are emitted, indicating proper state tracking.")
        if not features["has_dangerous_call"] and not features["has_low_level_call"]:
            lines.append("- **No dangerous calls**: No low-level `.call()` detected — "
                         "the contract uses safe transfer methods.")
        if features["num_external_calls"] == 0:
            lines.append("- **No external calls**: No external calls detected, so reentrancy is not applicable.")
        if features["num_modifiers_applied"] > 0:
            lines.append(f"- **Modifiers applied**: {features['num_modifiers_applied']} modifier application(s) "
                         "on functions provide access control or reentrancy protection.")
        if features["num_require_assert"] > 0:
            lines.append(f"- **Input validation**: {features['num_require_assert']} `require`/`assert` statement(s) "
                         "provide runtime checks.")

    return "\n".join(lines)


# ── Routes ──────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    # Handle file upload or pasted code
    code = ""
    filename = "pasted_code.sol"

    if "file" in request.files and request.files["file"].filename:
        f = request.files["file"]
        filename = f.filename
        if not filename.endswith(".sol"):
            return jsonify({"error": "Please upload a .sol file"}), 400
        code = f.read().decode("utf-8", errors="replace")
    elif request.form.get("code"):
        code = request.form["code"]
    else:
        return jsonify({"error": "No file or code provided"}), 400

    if not code.strip():
        return jsonify({"error": "Empty file/code"}), 400

    # Extract features (Slither if available, else regex fallback)
    features, used_slither, slither_raw = extract_features_with_slither(code)

    # Build feature vector in correct order
    X = np.array([[features[c] for c in feature_cols]])
    X_s = scaler.transform(X)

    # Predict
    prediction = int(model.predict(X_s)[0])
    probabilities = model.predict_proba(X_s)[0]
    confidence = float(probabilities[prediction])

    # Store original ML result
    ml_prediction = prediction
    ml_confidence = confidence
    ml_label = "Reentrant (Vulnerable)" if ml_prediction == 1 else "Safe"

    # Note: no override hack — Slither features (has_reentrancy_eth, cei_violation)
    # are fed directly into the ML model, so the model should handle this correctly.
    slither_override = False

    # Generate explanations (rule-based + LLM)
    rule_explanation, llm_explanation = generate_explanations(features, prediction, confidence, code)

    # Build Slither-only verdict for comparison
    slither_reentrancy = [d for d in slither_raw["detectors"] if d["check"] in REENTRANCY_CHECKS]
    slither_verdict = "Not Run"
    if slither_raw["ran"]:
        slither_verdict = "Vulnerable" if slither_reentrancy else "No Finding"

    return jsonify({
        "filename": filename,
        "prediction": "Reentrant (Vulnerable)" if prediction == 1 else "Safe",
        "prediction_code": prediction,
        "confidence": round(confidence * 100, 1),
        "ml_prediction": ml_label,
        "ml_confidence": round(ml_confidence * 100, 1),
        "slither_override": slither_override,
        "features": features,
        "explanation": rule_explanation,
        "llm_explanation": llm_explanation,
        "analysis_mode": "Slither + ML" if used_slither else "Source Analysis + ML",
        "slither_available": SLITHER_BIN is not None,
        "slither_used": used_slither,
        "slither_raw": slither_raw,
        "slither_verdict": slither_verdict,
        "slither_reentrancy_findings": slither_reentrancy,
        "slither_all_findings": slither_raw["detectors"],
    })


if __name__ == "__main__":
    print("Starting Reentrancy Detection Web Demo...")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True, host="127.0.0.1", port=5000)
