#!/usr/bin/env python3
"""compare_results.py — Compare OpenClaw vs Ouroboros benchmark results."""
import json, sys
from pathlib import Path

def load(p): return json.loads(Path(p).read_text())

def compare(a_path, b_path):
    a, b = load(a_path), load(b_path)
    label_a = a["metadata"].get("agent", "openclaw")
    label_b = b["metadata"].get("agent", "ouroboros")
    model   = a["metadata"].get("model", "?")
    print(f"\n{'═'*68}")
    print(f"  {label_a.upper()} vs {label_b.upper()} | model={model}")
    print(f"{'═'*68}")
    print(f"  {'Task':<32} {label_a:>12} {label_b:>12} {'Δ':>8}")
    print(f"  {'─'*66}")
    a_by = {t["task_id"]: t for t in a["tasks"]}
    b_by = {t["task_id"]: t for t in b["tasks"]}
    for tid in sorted(set(a_by)|set(b_by)):
        sa = a_by[tid]["grading"]["mean"] if tid in a_by else None
        sb = b_by[tid]["grading"]["mean"] if tid in b_by else None
        as_ = f"{sa:.2f}" if sa is not None else "N/A"
        bs_ = f"{sb:.2f}" if sb is not None else "N/A"
        if sa and sb:
            d = sb - sa; ds = f"{d:+.2f}"; flag = " ⬆" if d>0.1 else (" ⬇" if d<-0.1 else "")
        else:
            ds, flag = "─", ""
        print(f"  {tid:<32} {as_:>12} {bs_:>12} {ds:>8}{flag}")
    print(f"  {'─'*66}")
    aa, ba = a["summary"]["average_score"], b["summary"]["average_score"]
    print(f"  {'AVERAGE':<32} {aa:>12.2f} {ba:>12.2f} {ba-aa:>+8.2f}")
    print(f"{'═'*68}")
    winner = label_a if aa > ba else (label_b if ba > aa else "Tie")
    print(f"\n  Winner: {winner.upper()} (margin: {abs(ba-aa):.2f})\n")

if len(sys.argv) != 3:
    print(f"Usage: python {sys.argv[0]} <results_a.json> <results_b.json>"); sys.exit(1)
compare(sys.argv[1], sys.argv[2])
