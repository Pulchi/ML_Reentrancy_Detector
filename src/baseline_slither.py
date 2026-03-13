"""Slither-only baseline – evaluate Slither as a standalone reentrancy classifier."""
from __future__ import annotations
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from tqdm import tqdm

from .dataset import load_all
from .slither_utils import slither_has_reentrancy
from .config import BASELINE_CSV, OUTPUTS_DIR


def run_baseline(limit: int = 0) -> pd.DataFrame:
    """Run Slither on every contract and compute classification metrics."""
    samples = load_all(limit)
    rows = []
    for s in tqdm(samples, desc="Slither baseline"):
        pred = int(slither_has_reentrancy(s.path, s.solc_version))
        rows.append({
            "file": s.path.name,
            "version": s.version_folder,
            "true_label": s.label,
            "slither_pred": pred,
        })

    df = pd.DataFrame(rows)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(BASELINE_CSV, index=False)

    y_true = df["true_label"]
    y_pred = df["slither_pred"]
    print("\n=== Slither Baseline ===")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1       : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\n" + classification_report(y_true, y_pred, target_names=["safe", "reentrant"], zero_division=0))
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    run_baseline(args.limit)
