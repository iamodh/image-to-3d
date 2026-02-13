# PLAN.md

## 개발 전략

- GPU 없이 개발 가능한 모듈(processor, validator, bg_remover)부터 TDD로 구현
- AI 추론(generator)은 Colab에서 별도 검증 후 연동
- 전체 파이프라인 확인 후 Docker로 패키징

---

## Phase 1: 코어 파이프라인 — CPU 모듈 우선

> 목표: GPU 없는 WSL 환경에서 개발/테스트 가능한 모듈 완성

- [x] TECHSPEC.md 작성
- [x] CLAUDE.md 작성
- [x] PLAN.md 작성
- [x] 프로젝트 구조 생성 (디렉토리, __init__.py, requirements.txt, .gitignore)
- [x] src/processor.py 구현
  - [x] 메시 수리 (normals, winding, holes)
  - [x] 중복 정점/면 제거
  - [x] 크기 조정 (target_height 기준)
  - [x] 바닥면 정렬 + 원점 중심 정렬
  - [x] tests/test_processor.py 작성
- [x] src/validator.py 구현
  - [x] watertight 체크
  - [x] 크기 체크 (축별 최소 1mm)
  - [x] 면 법선 방향 체크
  - [x] 부피 계산
  - [x] tests/test_validator.py 작성
- [x] src/bg_remover.py 구현
  - [x] rembg로 배경 제거
  - [x] RGBA → 흰 배경 RGB 변환
  - [x] tests/test_bg_remover.py 작성
- [x] convert.py CLI 프레임 구현
  - [x] argparse 옵션 정의
  - [x] 파이프라인 오케스트레이션 (generator는 mock)
  - [x] 진행 상태 출력 ([1/4], [2/4]...)
- [x] tests/fixtures/sample_mesh.stl 생성 (테스트용)

**완료 조건**: `pytest tests/ -v` 전체 통과, generator mock 상태에서 CLI 실행 가능

---

## Phase 2: AI 추론 연동

> 목표: TripoSR 연동 + 전체 파이프라인 end-to-end 동작

- [ ] src/generator.py 구현
  - [x] TripoSR 모델 로드 (캐싱)
  - [x] 이미지 → 3D 메시 생성
  - [x] GPU 미감지 시 명확한 에러 메시지
- [ ] Colab에서 통합 테스트
  - [ ] setup_colab.sh 작성
  - [ ] notebooks/quickstart.ipynb 완성
  - [ ] 샘플 이미지 3종 이상 테스트
- [ ] 전체 파이프라인 검증
  - [ ] 이미지 입력 → STL 출력 end-to-end
  - [ ] 출력 STL을 슬라이서(Cura)에서 열어 프린팅 가능 여부 확인
- [ ] examples/ 디렉토리에 샘플 입력/출력 추가

**완료 조건**: Colab에서 `python convert.py -i sample.png -o output.stl` 실행하여 프린팅 가능한 STL 생성

---

## Phase 3: Docker 패키징 + 배포

> 목표: 어떤 환경에서든 동일하게 실행 가능한 Docker 이미지

- [ ] Dockerfile 작성
  - [ ] nvidia/cuda 베이스 이미지
  - [ ] TripoSR + 의존성 설치
  - [ ] ENTRYPOINT로 convert.py 실행
- [ ] docker-compose.yml 작성
- [ ] .dockerignore 작성
- [ ] 로컬 Docker 빌드 + 실행 테스트
- [ ] 클라우드 배포 테스트
  - [ ] Docker Hub에 이미지 푸시
  - [ ] RunPod 또는 Vast.ai에서 실행 확인
- [ ] README.md 작성 (설치/사용법 안내)

**완료 조건**: `docker run --gpus all image-to-3d -i input.png -o output.stl` 로 정상 동작

---

## 백로그 (Phase 3 이후)

- [ ] 모델 교체 옵션 (Trellis, InstantMesh 등)
- [ ] 배치 처리 (디렉토리 내 모든 이미지 일괄 변환)
- [ ] 자동 서포트 분석 (프린팅 방향 추천)
- [ ] 텍스처 매핑 (컬러 3D 프린팅, 3MF 포맷)
- [ ] 멀티뷰 입력 지원
- [ ] FastAPI 웹 서비스화
