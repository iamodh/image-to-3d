#!/usr/bin/env python3
"""Image-to-3D printable model CLI."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

from src import bg_remover, generator, processor, validator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="단일 이미지를 3D 프린터용 STL 파일로 변환")
    parser.add_argument("-i", "--input", required=True, help="입력 이미지 경로")
    parser.add_argument("-o", "--output", default="./output.stl", help="출력 파일 경로")
    parser.add_argument("--height", type=float, default=100.0, help="목표 높이 (mm)")
    parser.add_argument("--format", choices=["stl", "obj", "3mf"], default="stl", help="출력 포맷")
    parser.add_argument("--no-bg-remove", action="store_true", help="배경 제거 건너뛰기")
    parser.add_argument("--mc-resolution", type=int, default=256, help="Marching Cubes 해상도")
    parser.add_argument("-v", "--verbose", action="store_true", help="상세 로그")
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    start_time = time.time()

    if not args.no_bg_remove:
        print("[1/4] 배경 제거 중...")
        image = bg_remover.remove_background(args.input)
    else:
        print("[1/4] 배경 제거 건너뜀")
        image = args.input

    print("[2/4] 3D 메시 생성 중... (GPU 사용)")
    raw_mesh = generator.generate_mesh(image, mc_resolution=args.mc_resolution)

    print("[3/4] 메시 후처리 중...")
    processed_mesh = processor.process_mesh(raw_mesh, target_height=args.height)

    print("[4/4] 검증 및 저장 중...")
    report = validator.validate_mesh(processed_mesh)
    print("\n=== 검증 결과 ===")
    print(f"  Watertight: {'✅' if report['watertight'] else '❌'}")
    print(f"  정점 수: {report['vertices']:,}")
    print(f"  면 수: {report['faces']:,}")
    print(f"  크기 (mm): {report['dimensions']}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_path = output_path.with_suffix(f".{args.format}")
    processed_mesh.export(str(save_path))

    elapsed = time.time() - start_time
    print(f"\n저장 완료: {save_path}")
    print(f"소요 시간: {elapsed:.1f}초")
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
