# Colab E2E 실행 가이드 (샘플 3종)

아래 순서대로 실행하면 Phase 2 검증 결과(`examples/colab_e2e_report.json`)가 생성됩니다.

## 1) 프로젝트 준비
```bash
git clone <YOUR_REPO_URL>
cd image-to-3d
bash setup_colab.sh
```

## 2) 입력 이미지 3개 준비
- `examples/inputs/sample1.png`
- `examples/inputs/sample2.png`
- `examples/inputs/sample3.png`

Colab에서 업로드 후 `examples/inputs/`로 이동해도 됩니다.

## 3) 멀티 이미지 end-to-end 실행
```bash
python scripts/colab_e2e.py \
  --manifest examples/colab_test_manifest.txt \
  --output-dir examples/outputs \
  --report examples/colab_e2e_report.json \
  --height 100 \
  --mc-resolution 256
```

## 4) 완료 기준
- `summary.all_passed == true`
- `examples/outputs/*.stl` 3개 생성
- (권장) STL을 Cura에서 열어 출력 가능 여부 확인

## 5) 결과 공유
아래 두 파일 내용을 공유하면 문서/플랜 완료 처리 가능:
- `examples/colab_e2e_report.json`
- Cura 확인 결과(가능/불가 + 간단 사유)
