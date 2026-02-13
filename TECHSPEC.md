# TECHSPEC: Image-to-3D Printable Model CLI

> 단일 이미지(피규어/캐릭터)를 3D 프린터용 STL 파일로 변환하는 CLI 프로그램

---

## 1. 프로젝트 개요

### 목적

사진 또는 일러스트 한 장을 입력하면, AI 모델을 통해 3D 메시를 생성하고, 3D 프린팅에 적합하도록 후처리한 STL 파일을 출력하는 CLI 도구.

### 사용 예시

```bash
python convert.py --input figure.png --output figure.stl --height 100
```

### 대상 입력물

- 피규어, 캐릭터, 조각상 등 유기적 형상의 단일 이미지
- PNG/JPG 포맷
- 배경이 깨끗하거나 단색인 이미지에서 최적 결과

---

## 2. 시스템 아키텍처

### 파이프라인

```
[입력 이미지]
    │
    ▼
[1. 배경 제거] ─── rembg
    │
    ▼
[2. 3D 메시 생성] ─── TripoSR (AI 모델)
    │
    ▼
[3. 후처리] ─── Trimesh
    │  ├─ 메시 수리 (normals, holes, winding)
    │  ├─ 크기 조정 (target height 기준)
    │  └─ 바닥면 정렬
    │
    ▼
[4. 검증] ─── watertight 체크, 최소 벽 두께 확인
    │
    ▼
[STL 파일 출력]
```

### 핵심 컴포넌트

| 컴포넌트 | 역할 | 라이브러리 |
|----------|------|-----------|
| Background Remover | 입력 이미지에서 배경 제거 | `rembg` |
| 3D Generator | 단일 이미지 → 3D 메시 생성 | `TripoSR` |
| Mesh Processor | 메시 수리, 스케일링, 정렬 | `trimesh` |
| Validator | 프린팅 적합성 검증 | `trimesh` |
| CLI Interface | 사용자 인터페이스 | `argparse` |

---

## 3. 기술 스택

### 런타임 환경

- **Python**: 3.10+
- **GPU**: CUDA 지원 GPU (최소 8GB VRAM) 또는 Google Colab T4
- **OS**: Linux (Colab/클라우드), macOS/Windows (로컬 GPU 있을 경우)

### 핵심 의존성

```
# AI 모델
TripoSR          # 단일 이미지 → 3D 메시
torch >= 2.0     # PyTorch (CUDA 지원)
torchvision

# 이미지 처리
rembg             # 배경 제거
Pillow            # 이미지 로드/변환

# 3D 메시 처리
trimesh           # 메시 로드, 수리, 변환, 내보내기
numpy

# CLI
argparse          # 커맨드라인 인터페이스
```

### GPU 환경 옵션

| 환경 | GPU | 비용 | 용도 |
|------|-----|------|------|
| Google Colab (무료) | T4 (16GB) | 무료 | 프로토타입, 테스트 |
| Google Colab Pro | T4/A100 | ~$10/월 | 안정적 사용 |
| Kaggle Notebooks | T4 x2 | 무료 (주 30시간) | 대안 무료 옵션 |
| RunPod | 다양 | ~$0.2-0.5/시간 | 자동화, 배치 처리 |
| Vast.ai | 다양 | ~$0.1-0.4/시간 | 저비용 클라우드 GPU |

---

## 4. CLI 인터페이스 설계

### 명령어

```bash
python convert.py [OPTIONS] --input <IMAGE_PATH>
```

### 옵션

| 플래그 | 타입 | 기본값 | 설명 |
|--------|------|--------|------|
| `--input`, `-i` | str | (필수) | 입력 이미지 경로 |
| `--output`, `-o` | str | `./output.stl` | 출력 STL 파일 경로 |
| `--height` | float | `100.0` | 출력 모델의 목표 높이 (mm) |
| `--format` | str | `stl` | 출력 포맷 (`stl`, `obj`, `3mf`) |
| `--no-bg-remove` | flag | `False` | 배경 제거 건너뛰기 |
| `--mc-resolution` | int | `256` | Marching Cubes 해상도 (높을수록 정밀) |
| `--verbose`, `-v` | flag | `False` | 상세 로그 출력 |

### 사용 예시

```bash
# 기본 사용
python convert.py -i character.png

# 높이 150mm, OBJ 포맷 출력
python convert.py -i figure.jpg -o figure.obj --height 150 --format obj

# 이미 배경이 제거된 이미지
python convert.py -i transparent_bg.png --no-bg-remove

# 고해상도 메시
python convert.py -i detailed_figure.png --mc-resolution 512
```

---

## 5. 핵심 구현

### 5.1 프로젝트 구조

```
image-to-3d/
├── convert.py          # CLI 엔트리포인트
├── src/
│   ├── __init__.py
│   ├── bg_remover.py   # 배경 제거
│   ├── generator.py    # TripoSR 3D 생성
│   ├── processor.py    # 메시 후처리
│   └── validator.py    # 프린팅 적합성 검증
├── Dockerfile          # Docker 이미지 설계도
├── docker-compose.yml  # Docker 실행 설정
├── .dockerignore       # Docker 빌드 제외 파일
├── requirements.txt
├── setup_colab.sh      # Colab 환경 셋업 스크립트
├── notebooks/
│   └── quickstart.ipynb  # Colab 퀵스타트 노트북
├── examples/
│   └── sample_input.png
├── output/
└── TECHSPEC.md
```

### 5.2 메인 파이프라인 (convert.py)

```python
#!/usr/bin/env python3
"""Image-to-3D Printable Model CLI"""

import argparse
import time
from pathlib import Path

from src.bg_remover import remove_background
from src.generator import generate_mesh
from src.processor import process_mesh
from src.validator import validate_mesh


def main():
    parser = argparse.ArgumentParser(
        description="단일 이미지를 3D 프린터용 STL 파일로 변환"
    )
    parser.add_argument("-i", "--input", required=True, help="입력 이미지 경로")
    parser.add_argument("-o", "--output", default="./output.stl", help="출력 파일 경로")
    parser.add_argument("--height", type=float, default=100.0, help="목표 높이 (mm)")
    parser.add_argument("--format", choices=["stl", "obj", "3mf"], default="stl")
    parser.add_argument("--no-bg-remove", action="store_true", help="배경 제거 건너뛰기")
    parser.add_argument("--mc-resolution", type=int, default=256, help="Marching Cubes 해상도")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    start_time = time.time()

    # Step 1: 배경 제거
    if not args.no_bg_remove:
        print("[1/4] 배경 제거 중...")
        image = remove_background(args.input)
    else:
        print("[1/4] 배경 제거 건너뜀")
        image = args.input

    # Step 2: 3D 메시 생성
    print("[2/4] 3D 메시 생성 중... (GPU 사용)")
    raw_mesh = generate_mesh(image, mc_resolution=args.mc_resolution)

    # Step 3: 후처리
    print("[3/4] 메시 후처리 중...")
    processed_mesh = process_mesh(raw_mesh, target_height=args.height)

    # Step 4: 검증 및 저장
    print("[4/4] 검증 및 저장 중...")
    report = validate_mesh(processed_mesh)
    print(f"\n=== 검증 결과 ===")
    print(f"  Watertight: {'✅' if report['watertight'] else '❌'}")
    print(f"  정점 수: {report['vertices']:,}")
    print(f"  면 수: {report['faces']:,}")
    print(f"  크기 (mm): {report['dimensions']}")

    # 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_mesh.export(str(output_path))

    elapsed = time.time() - start_time
    print(f"\n✅ 저장 완료: {output_path}")
    print(f"⏱  소요 시간: {elapsed:.1f}초")


if __name__ == "__main__":
    main()
```

### 5.3 배경 제거 (src/bg_remover.py)

```python
"""이미지 배경 제거 모듈"""

from PIL import Image
from rembg import remove
from pathlib import Path
import io


def remove_background(image_path: str) -> Image.Image:
    """
    입력 이미지에서 배경을 제거하고 RGBA 이미지를 반환.
    TripoSR은 배경이 깨끗한 이미지에서 훨씬 좋은 결과를 냄.
    """
    input_image = Image.open(image_path)
    output_image = remove(input_image)

    # 흰 배경으로 변환 (TripoSR 입력 포맷)
    white_bg = Image.new("RGBA", output_image.size, (255, 255, 255, 255))
    white_bg.paste(output_image, mask=output_image.split()[3])

    return white_bg.convert("RGB")
```

### 5.4 3D 메시 생성 (src/generator.py)

```python
"""TripoSR 기반 3D 메시 생성 모듈"""

import torch
import trimesh
from PIL import Image
from tsr.system import TSR


_model = None  # 모델 캐싱


def _load_model():
    global _model
    if _model is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        _model.renderer.set_chunk_size(8192)
        _model.to(device)
    return _model


def generate_mesh(image, mc_resolution: int = 256) -> trimesh.Trimesh:
    """
    단일 이미지에서 3D 메시를 생성.

    Args:
        image: PIL Image 또는 이미지 파일 경로
        mc_resolution: Marching Cubes 해상도 (높을수록 정밀, 느림)

    Returns:
        trimesh.Trimesh 객체
    """
    model = _load_model()

    if isinstance(image, str):
        image = Image.open(image)

    with torch.no_grad():
        scene_codes = model([image], device=model.device)
        meshes = model.extract_mesh(scene_codes, resolution=mc_resolution)

    # TripoSR 출력을 trimesh로 변환
    mesh = meshes[0]
    vertices = mesh.vertices.cpu().numpy()
    faces = mesh.faces.cpu().numpy()

    return trimesh.Trimesh(vertices=vertices, faces=faces)
```

### 5.5 메시 후처리 (src/processor.py)

```python
"""3D 프린팅용 메시 후처리 모듈"""

import trimesh
import numpy as np


def process_mesh(mesh: trimesh.Trimesh, target_height: float = 100.0) -> trimesh.Trimesh:
    """
    AI 생성 메시를 3D 프린팅 가능하도록 후처리.

    Args:
        mesh: 원본 trimesh 메시
        target_height: 최종 모델 높이 (mm)

    Returns:
        후처리된 trimesh.Trimesh 객체
    """
    # 1. 메시 수리
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fill_holes(mesh)

    # 2. 중복 정점/면 제거
    mesh.merge_vertices()
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()

    # 3. 크기 조정 (target_height 기준, mm 단위)
    current_height = mesh.extents.max()  # 가장 긴 축 기준
    if current_height > 0:
        scale_factor = target_height / current_height
        mesh.apply_scale(scale_factor)

    # 4. 바닥면 정렬 (프린트 베드에 놓이도록 Z 최솟값을 0으로)
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])

    # 5. 원점 중심 정렬
    centroid = mesh.centroid.copy()
    centroid[2] = 0  # Z축은 이미 바닥 정렬했으므로 유지
    mesh.apply_translation(-centroid)

    return mesh
```

### 5.6 검증 (src/validator.py)

```python
"""3D 프린팅 적합성 검증 모듈"""

import trimesh
import numpy as np


def validate_mesh(mesh: trimesh.Trimesh) -> dict:
    """
    메시가 3D 프린팅에 적합한지 검증하고 리포트를 반환.

    Returns:
        dict: 검증 결과
            - watertight: bool (완전히 닫힌 메시인지)
            - vertices: int (정점 수)
            - faces: int (면 수)
            - dimensions: str (XYZ 크기, mm)
            - volume: float (부피, mm³)
            - issues: list[str] (발견된 문제)
    """
    issues = []

    # Watertight 체크
    is_watertight = mesh.is_watertight
    if not is_watertight:
        issues.append("메시가 watertight하지 않음 (구멍이 있을 수 있음)")

    # 크기 체크
    dims = mesh.extents
    for i, axis in enumerate(["X", "Y", "Z"]):
        if dims[i] < 1.0:
            issues.append(f"{axis}축 크기가 1mm 미만 ({dims[i]:.2f}mm)")

    # 면 방향 체크
    if mesh.face_normals is not None:
        inverted = np.sum(mesh.face_normals[:, 2] < 0) / len(mesh.face_normals)
        if inverted > 0.6:
            issues.append("면 법선이 대부분 뒤집혀 있을 수 있음")

    # 부피 계산 (watertight일 때만 의미 있음)
    volume = mesh.volume if is_watertight else -1

    return {
        "watertight": is_watertight,
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "dimensions": f"{dims[0]:.1f} x {dims[1]:.1f} x {dims[2]:.1f} mm",
        "volume": round(volume, 2),
        "issues": issues,
    }
```

---

## 6. Google Colab 셋업

### setup_colab.sh

```bash
#!/bin/bash
# Google Colab 환경 셋업 스크립트

set -e

echo "=== Image-to-3D Colab Setup ==="

# 1. TripoSR 클론
if [ ! -d "TripoSR" ]; then
    git clone https://github.com/VAST-AI-Research/TripoSR.git
fi
cd TripoSR

# 2. 의존성 설치
pip install -q -r requirements.txt
pip install -q rembg trimesh

# 3. GPU 확인
python -c "
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'✅ GPU: {gpu} ({mem:.1f}GB)')
else:
    print('❌ GPU를 찾을 수 없습니다. 런타임 → 런타임 유형 변경 → GPU 선택')
    exit(1)
"

echo "=== 셋업 완료 ==="
```

### Colab 퀵스타트 노트북 (notebooks/quickstart.ipynb)

아래 내용을 Colab에서 셀 단위로 실행:

```python
# --- Cell 1: 환경 셋업 ---
!git clone https://github.com/VAST-AI-Research/TripoSR.git
%cd TripoSR
!pip install -q -r requirements.txt
!pip install -q rembg trimesh

# GPU 확인
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "❌ GPU 없음")
```

```python
# --- Cell 2: 이미지 업로드 ---
from google.colab import files
from PIL import Image
from rembg import remove

uploaded = files.upload()  # 파일 선택 대화상자
input_filename = list(uploaded.keys())[0]

# 배경 제거
original = Image.open(input_filename)
no_bg = remove(original)

# 미리보기 (배경 제거 전후 비교)
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(original); ax1.set_title("원본"); ax1.axis("off")
ax2.imshow(no_bg); ax2.set_title("배경 제거"); ax2.axis("off")
plt.show()

# 흰 배경으로 변환
white_bg = Image.new("RGBA", no_bg.size, (255, 255, 255, 255))
white_bg.paste(no_bg, mask=no_bg.split()[3])
processed_image = white_bg.convert("RGB")
```

```python
# --- Cell 3: 3D 메시 생성 ---
import time
from tsr.system import TSR

model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)
model.renderer.set_chunk_size(8192)
model.to("cuda:0")

start = time.time()
with torch.no_grad():
    scene_codes = model([processed_image], device="cuda:0")
    meshes = model.extract_mesh(scene_codes, resolution=256)

elapsed = time.time() - start
print(f"✅ 3D 메시 생성 완료 ({elapsed:.1f}초)")
```

```python
# --- Cell 4: 후처리 + STL 저장 ---
import trimesh
import numpy as np

# trimesh로 변환
raw = meshes[0]
mesh = trimesh.Trimesh(
    vertices=raw.vertices.cpu().numpy(),
    faces=raw.faces.cpu().numpy(),
)

# 메시 수리
trimesh.repair.fix_normals(mesh)
trimesh.repair.fix_winding(mesh)
trimesh.repair.fill_holes(mesh)
mesh.merge_vertices()
mesh.remove_degenerate_faces()

# 크기 조정 (높이 100mm)
TARGET_HEIGHT = 100  # mm, 원하는 크기로 변경
scale = TARGET_HEIGHT / mesh.extents.max()
mesh.apply_scale(scale)

# 바닥 정렬
mesh.apply_translation([0, 0, -mesh.bounds[0][2]])

# 검증
print(f"Watertight: {'✅' if mesh.is_watertight else '❌'}")
print(f"정점: {len(mesh.vertices):,} / 면: {len(mesh.faces):,}")
print(f"크기: {mesh.extents[0]:.1f} x {mesh.extents[1]:.1f} x {mesh.extents[2]:.1f} mm")

# STL 저장
output_path = "figure_printable.stl"
mesh.export(output_path)
print(f"\n✅ 저장: {output_path}")
```

```python
# --- Cell 5: 3D 미리보기 + 다운로드 ---
# 간단한 3D 뷰 (matplotlib)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(
    mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
    triangles=mesh.faces, alpha=0.7, edgecolor='none'
)
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
plt.title("3D Preview")
plt.show()

# 다운로드
files.download(output_path)
```

---

## 7. Docker 컨테이너화

### 개요

Docker는 "프로젝트가 돌아가는 환경 자체"를 통째로 패키징하는 도구다. Python, CUDA, TripoSR, 모든 pip 패키지가 하나의 이미지로 묶이기 때문에 어떤 머신에서든 동일하게 실행된다.

> pip/bundler가 "필요한 라이브러리 목록"을 관리한다면, Docker는 "라이브러리가 설치된 컴퓨터 자체"를 복제하는 것이다.

### 도입 시점

| 단계 | Docker 필요 여부 | 이유 |
|------|:---:|------|
| Colab 프로토타입 | ❌ | Colab이 환경을 제공하므로 불필요 |
| 로컬 GPU에서 반복 실행 | ⚠️ 선택 | CUDA/PyTorch 버전 충돌 방지에 유용 |
| RunPod/Vast.ai 배포 | ✅ 필수 | Docker 이미지를 올려서 실행하는 구조 |
| 다른 사람에게 배포 | ✅ 필수 | `docker run` 한 줄로 실행 가능 |
| 배치 처리 자동화 | ✅ 필수 | 재현 가능한 실행 환경 보장 |

**권장 순서: Colab에서 파이프라인 완성 → 정상 동작 확인 → Docker로 패키징**

### 핵심 개념

```
[Dockerfile]  →  docker build  →  [Image]  →  docker run  →  [Container]
 (설계도)                         (빌드된 환경)                (실행 인스턴스)
```

- **Dockerfile**: 환경 구성 레시피. "Ubuntu 깔고, Python 깔고, pip install 하고..." 를 코드로 기술
- **Image**: Dockerfile을 빌드한 결과물. 스냅샷처럼 저장되고 공유 가능 (Docker Hub 등)
- **Container**: Image를 실행한 인스턴스. 실제 프로세스가 돌아가는 격리된 환경

### Dockerfile

```dockerfile
# ==============================================================
# Stage 1: CUDA + Python 베이스
# ==============================================================
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 시스템 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python 기본 설정
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

# ==============================================================
# Stage 2: 의존성 설치
# ==============================================================
WORKDIR /app

# TripoSR 클론 + 의존성
RUN git clone https://github.com/VAST-AI-Research/TripoSR.git /app/TripoSR
RUN pip install --no-cache-dir -r /app/TripoSR/requirements.txt
RUN pip install --no-cache-dir rembg trimesh

# ==============================================================
# Stage 3: 프로젝트 코드
# ==============================================================
COPY src/ /app/src/
COPY convert.py /app/convert.py

# 입출력 디렉토리
RUN mkdir -p /app/input /app/output

# ==============================================================
# 실행
# ==============================================================
ENTRYPOINT ["python", "convert.py"]
CMD ["--help"]
```

### 사용법

```bash
# 1. 이미지 빌드 (최초 1회, 수 분 소요)
docker build -t image-to-3d .

# 2. 기본 실행
#    -v 옵션: 로컬 폴더 ↔ 컨테이너 폴더 연결
#    --gpus all: GPU 패스스루
docker run --gpus all \
    -v $(pwd)/input:/app/input \
    -v $(pwd)/output:/app/output \
    image-to-3d \
    -i /app/input/figure.png -o /app/output/model.stl --height 100

# 3. 높이/포맷 변경
docker run --gpus all \
    -v $(pwd)/input:/app/input \
    -v $(pwd)/output:/app/output \
    image-to-3d \
    -i /app/input/character.jpg -o /app/output/character.obj \
    --height 150 --format obj

# 4. 디버깅: 컨테이너 안에 직접 접속
docker run --gpus all -it --entrypoint /bin/bash \
    -v $(pwd)/input:/app/input \
    -v $(pwd)/output:/app/output \
    image-to-3d
```

### docker-compose.yml (선택)

자주 사용하는 옵션을 파일로 저장해두면 편하다:

```yaml
version: "3.8"
services:
  converter:
    build: .
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./input:/app/input
      - ./output:/app/output
```

```bash
# docker-compose로 실행
docker compose run converter -i /app/input/figure.png -o /app/output/model.stl
```

### 클라우드 GPU 배포 (RunPod/Vast.ai)

Docker 이미지를 만들어두면 클라우드 GPU에 바로 올릴 수 있다:

```bash
# 1. Docker Hub에 이미지 푸시
docker tag image-to-3d username/image-to-3d:latest
docker push username/image-to-3d:latest

# 2. RunPod에서 실행
#    RunPod 콘솔 → Custom Template → Docker Image에 "username/image-to-3d:latest" 입력
#    또는 RunPod CLI:
runpodctl create pod --gpuType "NVIDIA RTX A4000" \
    --imageName "username/image-to-3d:latest"
```

### .dockerignore

빌드 시 불필요한 파일을 제외한다:

```
.git
__pycache__
*.pyc
output/
input/
notebooks/
*.md
.env
```

### 프로젝트 구조 (Docker 포함)

```
image-to-3d/
├── convert.py
├── src/
│   ├── __init__.py
│   ├── bg_remover.py
│   ├── generator.py
│   ├── processor.py
│   └── validator.py
├── Dockerfile              # ← 추가
├── docker-compose.yml      # ← 추가
├── .dockerignore            # ← 추가
├── requirements.txt
├── setup_colab.sh
├── notebooks/
│   └── quickstart.ipynb
├── input/                   # 로컬 입력 이미지 (Git 제외)
├── output/                  # 결과 STL 파일 (Git 제외)
└── TECHSPEC.md
```

---

## 8. 알려진 제약 및 주의사항

### AI 모델 한계

- 이미지에 보이지 않는 뒷면은 AI가 추측하여 생성하므로, 복잡한 뒷면 디테일은 부정확할 수 있음
- 매우 얇거나 날카로운 부분(검, 날개, 머리카락 등)은 메시가 깨지기 쉬움
- 반투명/반사 재질의 물체는 결과가 좋지 않음

### 3D 프린팅 제약

- AI 생성 메시는 watertight하지 않은 경우가 빈번 → 슬라이서에서 자동 수리 기능 활용
- 가늘고 긴 부분은 서포트 구조가 필요하므로 슬라이서에서 서포트 설정 필수
- FDM 프린터 기준 최소 벽 두께 약 0.8mm (노즐 0.4mm 기준)

### 입력 이미지 최적화 팁

- 배경이 단순할수록 결과가 좋음 (단색 배경 권장)
- 물체가 이미지 중앙에, 전체의 70-80%를 차지하도록
- 조명이 고른 이미지가 좋음 (강한 그림자 피할 것)
- 해상도 512x512 이상 권장

---

## 9. 향후 확장 가능성

| 기능 | 설명 | 우선순위 |
|------|------|---------|
| 모델 교체 | Trellis, InstantMesh 등 최신 모델로 교체 | 높음 |
| Docker 패키징 | Colab 검증 후 Docker 이미지로 패키징 → 클라우드 배포 | 높음 |
| 배치 처리 | 여러 이미지를 한 번에 변환 | 중간 |
| 자동 서포트 | 프린팅 방향 + 서포트 위치 자동 추천 | 중간 |
| 텍스처 출력 | 컬러 3D 프린팅용 텍스처 매핑 (3MF) | 낮음 |
| 멀티뷰 입력 | 여러 각도 사진으로 정밀도 향상 | 낮음 |
| API 서비스화 | FastAPI로 HTTP 엔드포인트 제공 | 낮음 |
