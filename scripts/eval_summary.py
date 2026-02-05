#!/usr/bin/env python3
import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, List


DEFAULT_BLOCK_PATTERNS = [
    "cloudflare",
    "attention required",
    "verify you are human",
    "checking your browser",
    "unusual traffic",
    "sorry, you have been blocked",
    "access denied",
    "captcha",
]


@dataclass(frozen=True)
class TaskResult:
    task_id: int
    status: str  # PASS/HALF/FAIL/ERROR
    score: float
    matched: Optional[int] = None
    total_eval: Optional[int] = None
    steps_taken: Optional[int] = None
    expected_steps: Optional[int] = None
    blocked: Optional[bool] = None


def _parse_result_line(line: str) -> Tuple[Optional[int], Optional[TaskResult]]:
    # Task id from config path
    m_id = re.search(r"batch_configs/(\d+)\.json", line)
    task_id = int(m_id.group(1)) if m_id else None

    if "[ERROR]" in line:
        if task_id is None:
            return None, None
        return task_id, TaskResult(task_id=task_id, status="ERROR", score=0.0)

    m = re.search(
        r"\[Result\] \((PASS|HALF|FAIL) with step: (\d+)\s*/\s*(\d+) score: ([0-9.]+)\)",
        line,
    )
    if not m or task_id is None:
        return task_id, None

    status = m.group(1)
    matched = int(m.group(2))
    total_eval = int(m.group(3))
    score = float(m.group(4))
    return task_id, TaskResult(
        task_id=task_id,
        status=status,
        score=score,
        matched=matched,
        total_eval=total_eval,
    )


def _count_steps_from_render(render_path: Path) -> Optional[int]:
    if not render_path.exists():
        return None
    text = render_path.read_text(errors="ignore")
    # Each step block includes raw_parsed_prediction/prev_action. This is a stable marker in this repo.
    c = text.count("raw_parsed_prediction")
    if c:
        return c
    c = text.count("prev_action")
    if c:
        return c
    return None


def _is_blocked(render_path: Path, patterns: Iterable[str]) -> Optional[bool]:
    if not render_path.exists():
        return None
    text = render_path.read_text(errors="ignore").lower()
    return any(p.lower() in text for p in patterns)


def _expected_steps_from_config(cfg_path: Path) -> Optional[int]:
    if not cfg_path.exists():
        return None
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception:
        return None
    # Prefer reference_task_length; fall back to number of evaluation checks.
    if isinstance(cfg.get("reference_task_length"), int) and cfg["reference_task_length"] > 0:
        return int(cfg["reference_task_length"])
    if isinstance(cfg.get("evaluation"), list) and len(cfg["evaluation"]) > 0:
        return int(len(cfg["evaluation"]))
    return None


def summarize(results: List[TaskResult]) -> dict:
    """
    SR: completed / total (PASS only)
    CR: succeeded checks / total checks (sum(matched)/sum(total_eval))
    efficiency: steps_taken / expected_steps (sum ratio), plus mean ratio for debugging
    """
    total = len(results)
    pass_cnt = sum(1 for r in results if r.status == "PASS")

    # Some runs log `score` per-task as matched/total_eval; keep both representations.
    score_sum = sum(r.score for r in results)  # ERROR already encoded as 0

    matched_sum = 0
    total_eval_sum = 0
    for r in results:
        if r.matched is None or r.total_eval is None:
            continue
        if r.total_eval <= 0:
            continue
        matched_sum += r.matched
        total_eval_sum += r.total_eval
    cr = (matched_sum / total_eval_sum) if total_eval_sum else 0.0

    # Efficiency
    eff_values = []
    steps_sum = 0
    expected_sum = 0
    for r in results:
        if r.steps_taken is None or r.expected_steps is None:
            continue
        if r.expected_steps <= 0:
            continue
        steps_sum += r.steps_taken
        expected_sum += r.expected_steps
        eff_values.append(r.steps_taken / r.expected_steps)
    eff_sum_ratio = (steps_sum / expected_sum) if expected_sum else 0.0
    eff_mean_ratio = sum(eff_values) / len(eff_values) if eff_values else 0.0

    counts = {}
    for k in ["PASS", "HALF", "FAIL", "ERROR"]:
        counts[k] = sum(1 for r in results if r.status == k)

    return {
        "total": total,
        "SR": (pass_cnt / total) if total else 0.0,
        "CR": cr,
        "avg_score": (score_sum / total) if total else 0.0,
        "efficiency": eff_sum_ratio,
        "efficiency_mean_ratio": eff_mean_ratio,
        "counts": counts,
        "eff_n": len(eff_values),
        "cr_n": total_eval_sum,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", type=str, required=True, help="A result directory containing result_0.txt and render_*.html files.")
    ap.add_argument("--result_file", type=str, default="result_0.txt")
    ap.add_argument("--fill_missing_from_batch_configs", action="store_true", help="Treat configs without a parsed result line as ERROR.")
    ap.add_argument("--block_pattern", action="append", default=[], help="Substring that marks a task as blocked (repeatable).")
    args = ap.parse_args()

    result_dir = Path(args.result_dir)
    result_file = result_dir / args.result_file
    if not result_file.exists():
        raise SystemExit(f"Missing {result_file}")

    patterns = args.block_pattern if args.block_pattern else DEFAULT_BLOCK_PATTERNS

    tasks = {}
    for line in result_file.read_text(errors="ignore").splitlines():
        tid, tr = _parse_result_line(line)
        if tid is None or tr is None:
            continue
        tasks[tid] = tr

    if args.fill_missing_from_batch_configs:
        bc_dir = result_dir / "batch_configs"
        if bc_dir.exists():
            for cfg in bc_dir.glob("*.json"):
                try:
                    tid = int(cfg.stem)
                except Exception:
                    continue
                if tid not in tasks:
                    tasks[tid] = TaskResult(task_id=tid, status="ERROR", score=0.0)

    enriched: list[TaskResult] = []
    for tid, tr in sorted(tasks.items(), key=lambda x: x[0]):
        render_path = result_dir / f"render_{tid}.html"
        cfg_path = result_dir / "batch_configs" / f"{tid}.json"
        steps_taken = _count_steps_from_render(render_path)
        expected_steps = _expected_steps_from_config(cfg_path)
        blocked = _is_blocked(render_path, patterns)
        enriched.append(
            TaskResult(
                task_id=tid,
                status=tr.status,
                score=tr.score,
                matched=tr.matched,
                total_eval=tr.total_eval,
                steps_taken=steps_taken,
                expected_steps=expected_steps,
                blocked=blocked,
            )
        )

    overall = summarize(enriched)
    # If a task is marked blocked but still reaches PASS, we count it as non-blocked for summary purposes.
    # This avoids false positives from pages that embed reCAPTCHA widgets without actually blocking access.
    non_blocked = [r for r in enriched if (r.blocked is False) or (r.status == "PASS")]
    non_blocked_summary = summarize(non_blocked)

    # Print a compact report.
    print(f"result_dir: {result_dir}")
    print(f"tasks: {overall['total']}")
    print(
        f"SR: {overall['SR']:.4f}  "
        f"CR: {overall['CR']:.4f} (checks={overall['cr_n']})  "
        f"efficiency: {overall['efficiency']:.4f} (n={overall['eff_n']})  "
        f"avg_score: {overall['avg_score']:.4f}"
    )
    print(f"counts: {overall['counts']}")
    print("")
    print("excluding_blocked:")
    print(f"tasks: {non_blocked_summary['total']}")
    print(
        f"SR: {non_blocked_summary['SR']:.4f}  "
        f"CR: {non_blocked_summary['CR']:.4f} (checks={non_blocked_summary['cr_n']})  "
        f"efficiency: {non_blocked_summary['efficiency']:.4f} (n={non_blocked_summary['eff_n']})  "
        f"avg_score: {non_blocked_summary['avg_score']:.4f}"
    )
    print(f"counts: {non_blocked_summary['counts']}")


if __name__ == "__main__":
    main()
