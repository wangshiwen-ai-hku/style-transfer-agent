#!/usr/bin/env bash
# Sync code/ to the GPU server.
#
#   bash sync_to_server.sh            # push code to the server
#   bash sync_to_server.sh --dry      # show what would transfer, change nothing
#   bash sync_to_server.sh --pull     # bring outputs/ and metrics/ back
#
# The link to this host is intermittent, so this retries with --partial rather
# than restarting from scratch, and keeps SSH keepalives on. A bare `rsync` over
# this link tends to die mid-transfer with "unexpected end of file".
# NOTE: no `set -u`. macOS ships bash 3.2, where "${ARR[@]}" on an empty array
# under `set -u` raises "unbound variable" -- which broke the --dry path here.
set -o pipefail

REMOTE="${REMOTE:-swwang@172.18.32.151}"
RPATH="${RPATH:-/homedata/swwang/projects/AgenticST}"
LOCAL="$(cd "$(dirname "$0")" && pwd)"
RETRIES="${RETRIES:-5}"

SSH_OPTS="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4"

# .env is excluded on purpose: the key should be placed on the server once, by
# hand, rather than re-copied on every sync (and never deleted by --delete).
#   scp .env $REMOTE:$RPATH/.env
EXCLUDES=(
  --exclude '__pycache__' --exclude '*.pyc' --exclude '.git' --exclude '.DS_Store'
  --exclude '.env'
  --exclude '*.pth' --exclude '*.safetensors' --exclude '*.ckpt'
  --exclude 'third_party/*/checkpoints'
  --exclude 'outputs/' --exclude 'result_*/' --exclude 'traces/'
)

MODE="push"; DRY=()
for a in "$@"; do
  case "$a" in
    --dry)  DRY=(-n) ;;
    --pull) MODE="pull" ;;
    *) echo "unknown option: $a" >&2; exit 1 ;;
  esac
done

run_rsync() {
  rsync -avz --partial --timeout=90 ${DRY[@]+"${DRY[@]}"} -e "$SSH_OPTS" "$@"
}

if [ "$MODE" = "push" ]; then
  echo "==> push  $LOCAL/  ->  $REMOTE:$RPATH/"
  SRC="$LOCAL/"; DST="$REMOTE:$RPATH/"
else
  echo "==> pull  $REMOTE:$RPATH/{outputs,metrics,traces}  ->  $LOCAL/"
  SRC="$REMOTE:$RPATH/"; DST="$LOCAL/"
  EXCLUDES=(--include 'outputs/***' --include 'metrics/***' --include 'traces/***'
            --include '*/' --exclude '*')
fi

for i in $(seq 1 "$RETRIES"); do
  echo "--- attempt $i/$RETRIES ---"
  if run_rsync "${EXCLUDES[@]}" "$SRC" "$DST"; then
    echo "==> done."
    exit 0
  fi
  echo "!! attempt $i failed; the link to this host drops intermittently. retrying in $((i * 5))s" >&2
  sleep $((i * 5))
done

echo "!! all $RETRIES attempts failed. Check VPN/network, then re-run; --partial means" >&2
echo "   the next run resumes rather than starting over." >&2
exit 1
