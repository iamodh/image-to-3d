# CLAUDE.md

## 프로젝트 개요

단일 이미지(피규어/캐릭터)를 3D 프린터용 STL 파일로 변환하는 CLI 도구.
상세 스펙은 TECHSPEC.md 참고.

## 기술 스택

- Python 3.10+, PyTorch, TripoSR, rembg, trimesh
- CLI: argparse
- 컨테이너: Docker (nvidia/cuda:11.8.0-runtime-ubuntu22.04 베이스)
- 테스트: pytest

## 프로젝트 구조

```
image-to-3d/
├── convert.py          # CLI 엔트리포인트 (파이프라인 오케스트레이션)
├── src/
│   ├── __init__.py
│   ├── bg_remover.py   # 배경 제거 (rembg)
│   ├── generator.py    # 3D 메시 생성 (TripoSR) — GPU 필수
│   ├── processor.py    # 메시 후처리 (trimesh) — CPU 전용
│   └── validator.py    # 프린팅 적합성 검증 — CPU 전용
├── tests/
│   ├── fixtures/
│   │   └── sample_mesh.stl   # 테스트용 샘플 메시
│   ├── test_processor.py
│   ├── test_validator.py
│   └── test_bg_remover.py
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
├── setup_colab.sh
├── notebooks/
│   └── quickstart.ipynb
├── input/               # 로컬 입력 이미지 (gitignore)
├── output/              # 결과 파일 (gitignore)
├── TECHSPEC.md
├── PLAN.md
└── CLAUDE.md
```

## 개발 규칙

### 아키텍처 원칙

- GPU 의존 코드(generator.py)와 CPU 전용 코드를 명확히 분리할 것
- 각 모듈(bg_remover, generator, processor, validator)은 독립적으로 import/실행 가능해야 함
- convert.py는 오케스트레이션만 담당하고, 비즈니스 로직은 src/ 모듈에 위치

### 코딩 컨벤션

- 타입 힌트 사용 (함수 시그니처에 파라미터 타입, 리턴 타입 명시)
- docstring 작성 (모듈, 클래스, public 함수)
- 3D 관련 단위는 mm 기준 통일
- 에러 메시지는 한국어로 작성

### GPU/CPU 분리 패턴

```python
# generator.py — GPU 필수 모듈
# 이 모듈은 CUDA GPU가 없으면 실행 불가
# 테스트 시 mock으로 대체

# processor.py, validator.py — CPU 전용 모듈
# GPU 없이 독립 테스트 가능
# trimesh만 의존
```

## 테스트

### GPU 없는 환경 (WSL 로컬)

```bash
# processor, validator, bg_remover 단위 테스트
pytest tests/ -v

# generator는 mock으로 대체
pytest tests/ -v -k "not gpu"
```

### GPU 있는 환경 (Colab / 클라우드)

```bash
# 전체 파이프라인 통합 테스트
python convert.py -i examples/sample_input.png -o output/test.stl -v
```

### 테스트 픽스처

- `tests/fixtures/sample_mesh.stl`: 정상적인 watertight 메시
- 테스트에서 generator.py의 출력을 시뮬레이션할 때 이 파일 사용

## CLI 인터페이스

```bash
python convert.py [OPTIONS] --input <IMAGE_PATH>

# 필수
--input, -i          입력 이미지 경로

# 선택
--output, -o         출력 파일 경로 (기본: ./output.stl)
--height             목표 높이 mm (기본: 100.0)
--format             출력 포맷: stl, obj, 3mf (기본: stl)
--no-bg-remove       배경 제거 건너뛰기
--mc-resolution      Marching Cubes 해상도 (기본: 256)
--verbose, -v        상세 로그
```

## 의존성 설치

```bash
# CPU 전용 개발 (WSL 로컬)
pip install rembg trimesh numpy Pillow pytest

# GPU 포함 (Colab / Docker)
pip install -r requirements.txt
```

## 현재 상태

Phase 1: 코어 파이프라인 개발 중 (CPU 모듈 우선)
