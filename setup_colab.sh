#!/usr/bin/env bash
set -euo pipefail

echo "=== Image-to-3D Colab Setup ==="

if [ ! -d "TripoSR" ]; then
  git clone https://github.com/VAST-AI-Research/TripoSR.git
fi
cd TripoSR

pip install -q -r requirements.txt
pip install -q rembg trimesh

python - <<'PY'
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu} ({mem:.1f}GB)")
else:
    raise SystemExit("GPU를 찾을 수 없습니다. Colab 런타임에서 GPU를 활성화하세요.")
PY

echo "=== Setup Complete ==="
