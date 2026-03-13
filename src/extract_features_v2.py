"""
Extract IMPROVED features for reentrancy detection (v2).
Adds ~12 new features on top of the original 10, targeting
specific misclassification patterns (FP and FN).

NEW features to reduce FP (safe contracts misclassified as ree):
  11. has_mutex_pattern       – bool-based mutex (require(!flag); flag=true; ... flag=false)
  12. has_block_protection    – uses block.number as protection
  13. num_require_assert      – count of require() / assert() statements
  14. has_state_set_before_call– state update BEFORE external call (CEI-compliant)
  15. num_modifiers_applied   – count of modifier usages on functions

NEW features to reduce FN (ree contracts misclassified as safe):
  16. has_internal_ext_call   – external call folded into internal function
  17. has_interface_cast      – I(addr).method() pattern (hidden external call)
  18. has_extcodesize         – uses extcodesize (ineffective protection)
  19. has_nested_mapping      – mapping of mapping (complex state)

General discriminative features:
  20. lines_of_code           – total lines of source
  21. has_msg_value           – uses msg.value
  22. has_payable             – has payable function
"""

import csv, json, re, glob
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent.parent
DATASET_ROOT = Path(r"C:\Users\vrasp\ml_reentrancy\manually-verified-reentrancy-dataset\dataset")

VERSIONS = {
    "04": {"sol": DATASET_ROOT / "0_4", "json": BASE / "outputs" / "json_04", "dot": BASE / "outputs" / "dot_04"},
    "05": {"sol": DATASET_ROOT / "0_5", "json": BASE / "outputs" / "json_05", "dot": BASE / "outputs" / "dot_05"},
    "08": {"sol": DATASET_ROOT / "0_8", "json": BASE / "outputs" / "json_08", "dot": BASE / "outputs" / "dot_08"},
}

REENTRANCY_CHECKS = {
    "reentrancy-eth", "reentrancy-no-eth", "reentrancy-benign",
    "reentrancy-unlimited-gas", "reentrancy-events",
}

FIELDNAMES = [
    "file", "version", "true_label",
    # Original 10
    "has_reentrancy_eth", "num_reentrancy_detectors",
    "cei_violation", "num_external_calls",
    "has_low_level_call", "has_dangerous_call",
    "num_functions", "has_modifier",
    "has_reentrancy_guard", "has_emit_after_call",
    # New 12
    "has_mutex_pattern", "has_block_protection",
    "num_require_assert", "has_state_set_before_call",
    "num_modifiers_applied",
    "has_internal_ext_call", "has_interface_cast",
    "has_extcodesize", "has_nested_mapping",
    "lines_of_code", "has_msg_value", "has_payable",
]

FEATURE_COLS = FIELDNAMES[3:]  # all feature columns


# ── helpers ──────────────────────────────────────────────────────────
def label_from_filename(fname):
    stem = Path(fname).stem
    if "_safe" in stem:
        return 0
    if "_ree" in stem:
        return 1
    raise ValueError(f"Cannot label: {fname}")


# ── JSON features ────────────────────────────────────────────────────
def json_features(json_path):
    has_eth = 0
    num_det = 0
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
        for d in data.get("results", {}).get("detectors", []):
            check = d.get("check", "")
            if check in REENTRANCY_CHECKS:
                num_det += 1
                if check == "reentrancy-eth":
                    has_eth = 1
    return {"has_reentrancy_eth": has_eth, "num_reentrancy_detectors": num_det}


# ── CFG / dot features ──────────────────────────────────────────────
def parse_dot(dot_path):
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
    return bool(re.search(r'\(->[a-zA-Z]', label))


def cfg_features(dot_dir, sol_name):
    pattern = str(dot_dir / f"{sol_name}-*-*.dot")
    dot_files = glob.glob(pattern)
    if not dot_files:
        pattern2 = str(dot_dir / f"{sol_name}*.dot")
        dot_files = [f for f in glob.glob(pattern2) if "call-graph" not in f]
    dot_files = [f for f in dot_files if "call-graph" not in f]

    cei = 0
    total_ext = 0
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


# ── Source-code features (original 6 + new 12) ──────────────────────
def source_features(sol_path):
    code = sol_path.read_text(encoding="utf-8", errors="replace")

    # --- Original 6 ---
    has_low_level = 1 if re.search(r'\.call', code) else 0
    uses_call = bool(re.search(r'\.call[\.\({\s]', code))
    has_dangerous = 1 if uses_call else 0
    num_funcs = len(re.findall(r'\bfunction\b', code))
    has_mod = 1 if re.search(r'\bmodifier\b', code) else 0
    has_guard = 1 if re.search(r'ReentrancyGuard|nonReentrant|noReentrant', code, re.IGNORECASE) else 0
    has_emit = 1 if re.search(r'\bemit\b', code) else 0

    # --- NEW: Mutex pattern ---
    # Detect bool-based mutex: bool variable + require(!var) + var = true/false
    has_bool_private = bool(re.search(r'\bbool\b.*\b\w+\b', code))
    has_require_not = bool(re.search(r'require\s*\(\s*!\s*\w+', code))
    has_mutex_pattern = 1 if (has_bool_private and has_require_not and not has_guard) else 0

    # --- NEW: Block protection ---
    has_block_protection = 1 if re.search(r'block\.number|block\.timestamp', code) else 0

    # --- NEW: require/assert count ---
    num_require_assert = len(re.findall(r'\b(?:require|assert)\s*\(', code))

    # --- NEW: State set before call ---
    # Look for pattern: variable = ... (state write) followed by .call/.send/.transfer
    # Simplified: find if there's an assignment before any external call in same function
    has_state_set_before_call = 0
    # Extract function bodies and check order
    func_bodies = re.findall(
        r'function\s+\w+[^{]*\{(.*?)\n\s*\}',
        code, re.DOTALL
    )
    for body in func_bodies:
        lines = body.split('\n')
        found_assignment = False
        for line in lines:
            stripped = line.strip()
            # State variable assignment (not local variable declaration)
            if re.search(r'\b\w+\s*\[.*\]\s*=', stripped) or re.search(r'\b\w+\s*=\s*(?!.*\bfunction\b)', stripped):
                if not re.search(r'\b(uint|int|bool|address|string|bytes|mapping)\b', stripped):
                    found_assignment = True
            # External call
            if found_assignment and re.search(r'\.(call|send|transfer)\s*[\.\({]', stripped, re.IGNORECASE):
                has_state_set_before_call = 1
                break

    # --- NEW: Count modifier applications ---
    # Modifiers applied in function signatures (words after ) and before {)
    modifier_defs = set(re.findall(r'modifier\s+(\w+)', code))
    num_modifiers_applied = 0
    for md in modifier_defs:
        num_modifiers_applied += len(re.findall(r'\b' + re.escape(md) + r'\b', code)) - 1  # -1 for definition

    # --- NEW: Internal function wrapping external call ---
    # Detect pattern: internal/private function that contains .call/.send/.transfer
    has_internal_ext_call = 0
    internal_funcs = re.findall(
        r'function\s+\w+[^{]*\b(?:internal|private)\b[^{]*\{(.*?)\n\s*\}',
        code, re.DOTALL
    )
    for body in internal_funcs:
        if re.search(r'\.(call|send|transfer)\s*[\.\({]', body, re.IGNORECASE):
            has_internal_ext_call = 1
            break
    # Also check if call is done through a named internal function
    if not has_internal_ext_call:
        # Find function calls to internal functions from withdraw-like functions
        internal_func_names = set(re.findall(
            r'function\s+(\w+)[^{]*\b(?:internal|private)\b', code
        ))
        public_bodies = re.findall(
            r'function\s+\w+[^{]*(?:public|external)[^{]*\{(.*?)\n\s*\}',
            code, re.DOTALL
        )
        for body in public_bodies:
            for fn in internal_func_names:
                if re.search(r'\b' + re.escape(fn) + r'\s*\(', body):
                    has_internal_ext_call = 1
                    break

    # --- NEW: Interface cast pattern ---
    # I(addr).method() or InterfaceName(address).method()
    has_interface_cast = 0
    interfaces = re.findall(r'\binterface\s+(\w+)', code)
    for iface in interfaces:
        if re.search(re.escape(iface) + r'\s*\([^)]*\)\s*\.', code):
            has_interface_cast = 1
            break
    # Also check general pattern: CapitalWord(address_var).method()
    if not has_interface_cast:
        if re.search(r'\b[A-Z]\w*\s*\(\s*\w+\s*\)\s*\.\s*\w+\s*\(', code):
            has_interface_cast = 1

    # --- NEW: extcodesize (ineffective reentrancy protection) ---
    has_extcodesize = 1 if re.search(r'extcodesize', code) else 0

    # --- NEW: Nested mapping ---
    has_nested_mapping = 1 if re.search(r'mapping\s*\([^)]*mapping\s*\(', code) else 0

    # --- NEW: Lines of code ---
    lines_of_code = len([l for l in code.split('\n') if l.strip()])

    # --- NEW: msg.value ---
    has_msg_value = 1 if re.search(r'msg\.value', code) else 0

    # --- NEW: payable ---
    has_payable = 1 if re.search(r'\bpayable\b', code) else 0

    return {
        "has_low_level_call": has_low_level,
        "has_dangerous_call": has_dangerous,
        "num_functions": num_funcs,
        "has_modifier": has_mod,
        "has_reentrancy_guard": has_guard,
        "has_emit_after_call": has_emit,
        # New features
        "has_mutex_pattern": has_mutex_pattern,
        "has_block_protection": has_block_protection,
        "num_require_assert": num_require_assert,
        "has_state_set_before_call": has_state_set_before_call,
        "num_modifiers_applied": num_modifiers_applied,
        "has_internal_ext_call": has_internal_ext_call,
        "has_interface_cast": has_interface_cast,
        "has_extcodesize": has_extcodesize,
        "has_nested_mapping": has_nested_mapping,
        "lines_of_code": lines_of_code,
        "has_msg_value": has_msg_value,
        "has_payable": has_payable,
    }


# ── main ─────────────────────────────────────────────────────────────
def process_version(ver, paths):
    sol_dir = paths["sol"]
    json_dir = paths["json"]
    dot_dir = paths["dot"]
    sol_files = sorted(sol_dir.glob("*.sol"))

    rows = []
    for sf in sol_files:
        fname = sf.name
        row = {
            "file": fname,
            "version": ver,
            "true_label": label_from_filename(fname),
        }
        row.update(json_features(json_dir / f"{sf.stem}.json"))
        row.update(cfg_features(dot_dir, fname))
        row.update(source_features(sf))
        rows.append(row)

    # Write per-version CSV
    out = BASE / "outputs" / f"features_v2_{ver}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)
    print(f"[{ver}] Wrote {len(rows)} rows to {out.name}")

    safe = [r for r in rows if r["true_label"] == 0]
    ree  = [r for r in rows if r["true_label"] == 1]
    print(f"      Safe={len(safe)}, Reentrant={len(ree)}")
    return rows


def main():
    all_rows = []
    for ver, paths in VERSIONS.items():
        all_rows.extend(process_version(ver, paths))

    # Combined CSV
    out_all = BASE / "outputs" / "features_v2_all.csv"
    with open(out_all, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nCombined: {len(all_rows)} rows -> {out_all.name}")

    # Quick feature distribution
    print("\n=== Feature distribution ===")
    for col in FEATURE_COLS:
        vals = [float(r[col]) for r in all_rows]
        safe_vals = [float(r[col]) for r in all_rows if r["true_label"] == 0]
        ree_vals  = [float(r[col]) for r in all_rows if r["true_label"] == 1]
        print(f"  {col:30s}  safe_avg={sum(safe_vals)/len(safe_vals):.3f}  ree_avg={sum(ree_vals)/len(ree_vals):.3f}")

if __name__ == "__main__":
    main()
