#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Check solutions in discrete_searcher output TXT files against discrete_searcher_sub constraints.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from Simulation.discrete_searcher_sub import OptimizedConfig, UnifiedLotteryOptimizer


RE_MODE = re.compile(r"Mode:\s*outcomes=(\d+),\s*stake=([a-zA-Z]+)")
RE_PARAMS = re.compile(
    r"Prospect theory parameters:\s*alpha=([0-9.]+),\s*lambda=([0-9.]+),\s*gamma=([0-9.]+)(?:,\s*r=([0-9.]+))?"
)
RE_VIOLATION = re.compile(r"Violation threshold:\s*([0-9.]+)")

RE_STAGE1 = re.compile(r"Stage 1:\s*b11=\s*(-?\d+),\s*b12=\s*(-?\d+)")
RE_STAGE2_4 = re.compile(r"Stage 2:\s*c21=\s*(-?\d+),\s*c22=\s*(-?\d+)")
RE_STAGE2_6 = re.compile(r"Stage 2:\s*c21=\s*(-?\d+),\s*c22=\s*(-?\d+),\s*c23=\s*(-?\d+)")
RE_STAGE3_4 = re.compile(
    r"Stage 3:\s*c31=\s*(-?\d+),\s*c32=\s*(-?\d+),\s*c33=\s*(-?\d+),\s*c34=\s*(-?\d+)"
)
RE_STAGE3_6 = re.compile(
    r"Stage 3:\s*c31=\s*(-?\d+),\s*c32=\s*(-?\d+),\s*c33=\s*(-?\d+),\s*c34=\s*(-?\d+),\s*c35=\s*(-?\d+),\s*c36=\s*(-?\d+)"
)
RE_PROBS_4 = re.compile(r"Probabilities:\s*p1=([0-9.]+),\s*p2=([0-9.]+),\s*p3=([0-9.]+)")
RE_PROBS_6 = re.compile(
    r"Probabilities:\s*p1=([0-9.]+),\s*p2=([0-9.]+),\s*p3=([0-9.]+),\s*p4=([0-9.]+),\s*p5=([0-9.]+)"
)


def _parse_mode(lines: List[str], fallback_name: str) -> Tuple[int, str]:
    for line in lines:
        match = RE_MODE.search(line)
        if match:
            return int(match.group(1)), match.group(2).strip().lower()
    name = fallback_name.lower()
    if "6outcomes" in name or "6 outcomes" in name:
        return 6, "lo"
    return 4, "lo"


def _parse_params(lines: List[str]) -> Tuple[float, float, float, float]:
    alpha = OptimizedConfig().alpha
    lambda_ = OptimizedConfig().lambda_
    gamma = OptimizedConfig().gamma
    r_value = OptimizedConfig().r
    for line in lines:
        match = RE_PARAMS.search(line)
        if match:
            alpha = float(match.group(1))
            lambda_ = float(match.group(2))
            gamma = float(match.group(3))
            if match.group(4) is not None:
                r_value = float(match.group(4))
            break
    return alpha, lambda_, gamma, r_value


def _parse_violation_threshold(lines: List[str]) -> float:
    for line in lines:
        match = RE_VIOLATION.search(line)
        if match:
            return float(match.group(1))
    return 1.0


def _parse_solutions(lines: List[str], outcomes: int) -> List[Dict[str, float]]:
    solutions: List[Dict[str, float]] = []
    current: Optional[Dict[str, float]] = None

    required_keys_4 = {"b11", "b12", "c21", "c22", "c31", "c32", "c33", "c34", "p1", "p2", "p3"}
    required_keys_6 = {
        "b11",
        "b12",
        "c21",
        "c22",
        "c23",
        "c31",
        "c32",
        "c33",
        "c34",
        "c35",
        "c36",
        "p1",
        "p2",
        "p3",
        "p4",
        "p5",
    }

    for line in lines:
        m_stage1 = RE_STAGE1.search(line)
        if m_stage1:
            current = {
                "b11": float(m_stage1.group(1)),
                "b12": float(m_stage1.group(2)),
            }
            continue

        if current is None:
            continue

        if outcomes == 4:
            m_stage2 = RE_STAGE2_4.search(line)
            if m_stage2:
                current["c21"] = float(m_stage2.group(1))
                current["c22"] = float(m_stage2.group(2))
            m_stage3 = RE_STAGE3_4.search(line)
            if m_stage3:
                current["c31"] = float(m_stage3.group(1))
                current["c32"] = float(m_stage3.group(2))
                current["c33"] = float(m_stage3.group(3))
                current["c34"] = float(m_stage3.group(4))
            m_probs = RE_PROBS_4.search(line)
            if m_probs:
                current["p1"] = float(m_probs.group(1))
                current["p2"] = float(m_probs.group(2))
                current["p3"] = float(m_probs.group(3))
        else:
            m_stage2 = RE_STAGE2_6.search(line)
            if m_stage2:
                current["c21"] = float(m_stage2.group(1))
                current["c22"] = float(m_stage2.group(2))
                current["c23"] = float(m_stage2.group(3))
            m_stage3 = RE_STAGE3_6.search(line)
            if m_stage3:
                current["c31"] = float(m_stage3.group(1))
                current["c32"] = float(m_stage3.group(2))
                current["c33"] = float(m_stage3.group(3))
                current["c34"] = float(m_stage3.group(4))
                current["c35"] = float(m_stage3.group(5))
                current["c36"] = float(m_stage3.group(6))
            m_probs = RE_PROBS_6.search(line)
            if m_probs:
                current["p1"] = float(m_probs.group(1))
                current["p2"] = float(m_probs.group(2))
                current["p3"] = float(m_probs.group(3))
                current["p4"] = float(m_probs.group(4))
                current["p5"] = float(m_probs.group(5))

        if outcomes == 4 and required_keys_4.issubset(current.keys()):
            solutions.append(current)
            current = None
        elif outcomes == 6 and required_keys_6.issubset(current.keys()):
            solutions.append(current)
            current = None

    return solutions


def _build_params(solution: Dict[str, float], outcomes: int) -> np.ndarray:
    if outcomes == 4:
        values = [
            solution["b11"],
            solution["b12"],
            solution["c21"],
            solution["c22"],
            solution["c31"],
            solution["c32"],
            solution["c33"],
            solution["c34"],
            solution["p1"],
            solution["p2"],
            solution["p3"],
        ]
    else:
        values = [
            solution["b11"],
            solution["b12"],
            solution["c21"],
            solution["c22"],
            solution["c23"],
            solution["c31"],
            solution["c32"],
            solution["c33"],
            solution["c34"],
            solution["c35"],
            solution["c36"],
            solution["p1"],
            solution["p2"],
            solution["p3"],
            solution["p4"],
            solution["p5"],
        ]
    return np.asarray(values, dtype=float)


def _write_report(path: Path, report_lines: List[str], output_path: Path) -> None:
    output_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote {output_path}")


def _check_file(path: Path, output_suffix: str = "_subcheck.txt") -> None:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    outcomes, stake = _parse_mode(lines, path.name)
    alpha, lambda_, gamma, r_value = _parse_params(lines)
    violation_threshold = _parse_violation_threshold(lines)
    solutions = _parse_solutions(lines, outcomes)

    config = OptimizedConfig(
        alpha=alpha,
        lambda_=lambda_,
        gamma=gamma,
        r=r_value,
        outcomes=outcomes,
        stake=stake,
        violation_threshold=violation_threshold,
        alt_params=[],
    )
    optimizer = UnifiedLotteryOptimizer(config)

    report_lines: List[str] = []
    report_lines.append("=" * 80)
    report_lines.append("Discrete Searcher Sub Constraint Check")
    report_lines.append("=" * 80)
    report_lines.append(f"Input file: {path}")
    report_lines.append(f"Mode: outcomes={outcomes}, stake={stake}")
    report_lines.append(
        f"Prospect theory parameters: alpha={alpha}, lambda={lambda_}, gamma={gamma}, r={r_value}"
    )
    report_lines.append(f"Violation threshold: {violation_threshold}")
    report_lines.append(f"Solutions parsed: {len(solutions)}")

    if not solutions:
        report_lines.append("No solutions parsed from file.")
        output_path = path.with_name(path.stem + output_suffix)
        _write_report(path, report_lines, output_path)
        return

    report_lines.append("")
    for idx, solution in enumerate(solutions, start=1):
        params = _build_params(solution, outcomes)
        violations, valid, _ = optimizer.check_full_constraints(params, track_stats=False)
        total = float(violations.get("total", 1e9))
        ok = bool(valid) and total < violation_threshold
        status = "OK" if ok else "FAIL"
        report_lines.append(f"Solution {idx}: {status} (total_violation={total:.6f})")
        if not ok:
            for key in sorted(k for k in violations.keys() if k != "total"):
                value = violations.get(key)
                if isinstance(value, (int, float)):
                    report_lines.append(f"  {key}={float(value):.6f}")
                else:
                    report_lines.append(f"  {key}={value}")
        report_lines.append("")

    output_path = path.with_name(path.stem + output_suffix)
    _write_report(path, report_lines, output_path)


def _collect_files(paths: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            continue
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.txt")))
    return files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check discrete_searcher TXT outputs against discrete_searcher_sub constraints."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Files or directories to scan (defaults to lottery_outcomes, lottery_results, lottery_results_6outcomes)",
    )
    args = parser.parse_args()

    paths = args.paths
    if not paths:
        defaults = []
        for name in ("lottery_outcomes", "lottery_results", "lottery_results_6outcomes"):
            if Path(name).exists():
                defaults.append(name)
        paths = defaults

    files = _collect_files(paths)
    if not files:
        print("No txt files found to check.")
        return 1

    for path in files:
        _check_file(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
