#!/usr/bin/env bash
# Set up the AgenticST environment on the GPU server.
#
#   bash setup_server.sh              # core only (enough to run AgenticST via APIYI)
#   bash setup_server.sh --preservation  # + face/geometry metrics (NO torch needed)
#   bash setup_server.sh --metrics    # + GPU baselines and FID/LPIPS (needs torch)
#   bash setup_server.sh --legacy     # + retired vendor SDKs (normally unnecessary)
#
# The server reaches api.apiyi.com but not Google/Aliyun/Volcengine, which is why
# the core requirements contain no vendor SDK. If a step here needs the open
# internet for PyPI and your server proxies it, export HTTPS_PROXY first.
set -euo pipefail

ENV_NAME="${ENV_NAME:-agenticst}"
PY_VERSION="${PY_VERSION:-3.11}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
WITH_METRICS=0; WITH_LEGACY=0; WITH_PRESERVATION=0
for a in "$@"; do
  case "$a" in
    --metrics) WITH_METRICS=1 ;;
    --preservation) WITH_PRESERVATION=1 ;;
    --legacy)  WITH_LEGACY=1 ;;
    *) echo "unknown option: $a" >&2; exit 1 ;;
  esac
done

echo "==> project root: $ROOT"

# ---------------------------------------------------------------- environment
# conda is often installed but absent from PATH in non-interactive shells, which
# would silently drop us into a venv built on the system python. On this server
# that is python 3.8, too old for the pinned langchain/pydantic/numpy, and the
# resulting pip resolution errors are hard to read. So look for conda properly.
if ! command -v conda >/dev/null 2>&1; then
  for c in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" /opt/conda; do
    if [ -x "$c/bin/conda" ]; then
      echo "==> found conda at $c (not on PATH; adding it)"
      export PATH="$c/bin:$PATH"
      break
    fi
  done
fi

if command -v conda >/dev/null 2>&1; then
  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "==> creating conda env '$ENV_NAME' (python $PY_VERSION)"
    conda create -y -n "$ENV_NAME" "python=$PY_VERSION"
  else
    echo "==> reusing conda env '$ENV_NAME'"
  fi
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
else
  echo "==> conda not found; using venv at $ROOT/.venv"
  [ -d "$ROOT/.venv" ] || python3 -m venv "$ROOT/.venv"
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi
python -V

# Fail early and legibly rather than letting pip produce an unreadable resolution
# error: langchain 1.x, pydantic 2.11 and numpy 2.x all require >= 3.10.
python - <<'PY'
import sys
if sys.version_info < (3, 10):
    sys.exit(
        f"\n!! python {sys.version.split()[0]} is too old.\n"
        "   requirements.txt needs >= 3.10 (langchain 1.x, pydantic 2.11, numpy 2.x).\n"
        "   Create a newer environment first, e.g.\n"
        "       conda create -y -n agenticst python=3.11 && conda activate agenticst\n"
        "   then re-run this script.\n")
PY

# ------------------------------------------------------------------ core deps
echo "==> installing core requirements"
python -m pip install --upgrade pip
python -m pip install -r "$ROOT/requirements.txt"

# --------------------------------------------------------------- metrics deps
if [ "$WITH_METRICS" = "1" ]; then
  echo "==> checking torch before installing metric packages"
  if ! python -c "import torch" >/dev/null 2>&1; then
    cat >&2 <<'MSG'
!! torch is not installed in this environment.
   Install the build that matches this server's CUDA version FIRST, e.g.
       pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   Letting pip resolve torch as a transitive dependency usually yields a CPU-only
   wheel, which will silently make every metric run take hours. Aborting.
MSG
    exit 1
  fi
  python - <<'PY'
import torch
print(f"==> torch {torch.__version__}, cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("!! WARNING: torch cannot see a GPU. Metric computation will run on CPU.")
PY
  python -m pip install -r "$ROOT/requirements-metrics.txt"
fi

# ---------------------------------------------- preservation metrics (no torch)
if [ "$WITH_PRESERVATION" = "1" ]; then
  echo "==> installing preservation-metric requirements (no torch required)"
  python -m pip install -r "$ROOT/requirements-preservation.txt"
  python - <<'PRESCHECK'
mods = {"mediapipe": "EAR / landmark drift / gaze",
        "insightface": "ID-Sim",
        "onnxruntime": "insightface backend",
        "skimage": "edge IoU"}
for m, what in mods.items():
    try:
        __import__(m)
        print(f"    ok      {m:14s} ({what})")
    except Exception as e:
        print(f"    MISSING {m:14s} ({what}) -- {e}")
PRESCHECK
fi

# ---------------------------------------------------------------- legacy deps
if [ "$WITH_LEGACY" = "1" ]; then
  echo "==> installing retired vendor SDKs"
  python -m pip install -r "$ROOT/requirements-legacy.txt"
fi

# ------------------------------------------------------------------ env check
echo
if [ ! -f "$ROOT/.env" ]; then
  cat >&2 <<MSG
!! $ROOT/.env is missing. It is intentionally excluded from rsync so the key is
   not copied around automatically. Create it with:

       APIYI_KEY=<your key>
       BASE_URL=https://api.apiyi.com/v1

   or copy it from your laptop:
       scp code/.env <server>:$ROOT/.env
MSG
else
  echo "==> found .env"
fi

echo
echo "==> verifying the APIYI endpoint end to end"
python "$ROOT/scripts/verify_apiyi.py" || {
  echo "!! verification failed -- fix this before launching any batch run." >&2
  exit 1
}

cat <<MSG

==> setup complete.

    Activate with:   conda activate $ENV_NAME    (or: source .venv/bin/activate)

    Smoke test one pair:
      python run_agent_limited.py --config config_limited.yaml \\
        --images "${AGENTICST_VERIFY_CONTENT:-compare/content/23.jpg}" "${AGENTICST_VERIFY_STYLE:-compare/style/114.jpg}" \\
        --result_dir result_rev_compare

    Then the 8 typical cases (auto-retries pairs with no output, then extracts traces):
      python run_compare_batch.py

    Preservation-contract ablation and its metrics:
      bash setup_server.sh --preservation
      python scripts/run_contract_ablation.py --repeats 10
      python scripts/compute_preservation_metrics.py --index ablation_contract/index.csv
MSG
