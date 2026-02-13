#!/usr/bin/env python3
"""Run end-to-end conversion for multiple images and save a JSON report.

Expected to run in a GPU-enabled environment with TripoSR importable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def run_one(image_path: Path, output_dir: Path, height: float, mc_resolution: int) -> dict:
    output_path = output_dir / f"{image_path.stem}.stl"
    cmd = [
        sys.executable,
        "convert.py",
        "-i",
        str(image_path),
        "-o",
        str(output_path),
        "--height",
        str(height),
        "--mc-resolution",
        str(mc_resolution),
        "--verbose",
    ]

    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = round(time.time() - started, 2)

    item = {
        "input": str(image_path),
        "output": str(output_path),
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-20:]),
        "output_exists": output_path.exists(),
        "output_size_bytes": output_path.stat().st_size if output_path.exists() else 0,
    }
    return item


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multi-image e2e conversion on Colab")
    parser.add_argument("--manifest", required=True, help="Text file with one image path per line")
    parser.add_argument("--output-dir", default="examples/outputs", help="Directory to store outputs")
    parser.add_argument("--report", default="examples/colab_e2e_report.json", help="JSON report path")
    parser.add_argument("--height", type=float, default=100.0)
    parser.add_argument("--mc-resolution", type=int, default=256)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    output_dir = Path(args.output_dir)
    report_path = Path(args.report)

    if not manifest.exists():
        raise SystemExit(f"manifest 파일이 없습니다: {manifest}")

    lines = [
        line.strip()
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if len(lines) < 3:
        raise SystemExit("샘플 이미지는 최소 3개 이상 필요합니다.")

    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for line in lines:
        image_path = Path(line)
        if not image_path.exists():
            results.append(
                {
                    "input": str(image_path),
                    "returncode": 999,
                    "error": "입력 파일이 존재하지 않습니다.",
                }
            )
            continue
        results.append(run_one(image_path, output_dir, args.height, args.mc_resolution))

    passed = [r for r in results if r.get("returncode") == 0 and r.get("output_exists")]
    failed = [r for r in results if r not in passed]

    summary = {
        "total": len(results),
        "passed": len(passed),
        "failed": len(failed),
        "all_passed": len(failed) == 0,
    }

    payload = {"summary": summary, "results": results}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False))
    print(f"report saved: {report_path}")

    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
