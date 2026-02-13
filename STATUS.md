# STATUS.md

기준 시점: 2026-02-13

## 프로젝트 상태 요약
- 현재 단계: Phase 1 (코어 파이프라인) 구현 완료
- 저장소 상태: clean working tree
- 최신 커밋: `404afa5` (`Bootstrap Phase 1 pipeline skeleton with tests`)

## 이번에 완료한 작업
- 에이전트 규칙 문서명 변경
  - `CLAUDE.md` -> `AGENTS.md`
- 프로젝트 기본 구조/의존성 파일 추가
  - `.gitignore`
  - `requirements.txt`
  - `src/__init__.py`
- 코어 모듈 구현
  - `src/bg_remover.py`
  - `src/processor.py`
  - `src/validator.py`
  - `src/generator.py` (Phase 2 연동 전 placeholder)
- CLI 프레임 구현
  - `convert.py`
- 테스트/픽스처 추가
  - `tests/test_bg_remover.py`
  - `tests/test_processor.py`
  - `tests/test_validator.py`
  - `tests/test_convert.py`
  - `tests/fixtures/sample_mesh.stl`
- 계획 문서 상태 업데이트
  - `PLAN.md` 체크리스트 반영

## 테스트 결과
실행 명령:
```bash
.venv/bin/python -m pytest -q
```
결과:
```text
6 passed in 0.43s
```

## 환경 관련 메모
- 시스템 Python은 PEP 668(externally-managed) 정책으로 전역 `pip install`이 제한됨
- 따라서 프로젝트 로컬 가상환경(`.venv`) 기반 실행을 표준 경로로 사용

## 현재 동작 범위
- CPU 기반 모듈(`bg_remover`, `processor`, `validator`) 테스트 완료
- `convert.py`는 파이프라인 오케스트레이션 동작 확인(테스트에서 mock 사용)
- `generator.py`는 아직 실제 TripoSR 추론 미연동(Phase 2 예정)

## 다음 작업(권장 순서)
1. Phase 2: `src/generator.py`에 TripoSR 실제 연동
2. Colab 통합 검증 스크립트/노트북 정리 (`setup_colab.sh`, `notebooks/quickstart.ipynb`)
3. 입력 이미지 -> 출력 STL end-to-end 검증
4. Phase 3: Dockerfile / docker-compose / 배포 검증

## 추가 진행 내역 (업데이트)
- `src/generator.py`에 TripoSR 지연 로딩/모델 캐싱/GPU 체크 구현 완료
- `tests/test_generator.py` 추가
- Colab 지원 파일 추가
  - `setup_colab.sh`
  - `notebooks/quickstart.ipynb`
- Docker 패키징 파일 추가
  - `Dockerfile`
  - `docker-compose.yml`
  - `.dockerignore`

## 현재 남은 블로커
- 이 WSL 세션에서 Docker 명령이 비활성화 상태
  - 메시지: WSL integration in Docker Desktop 비활성
  - 영향: `docker build`, `docker run` 로컬 검증 미실행
