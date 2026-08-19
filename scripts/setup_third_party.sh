#!/usr/bin/env bash
# Fetch the baseline systems this paper compares against, at the commits we used,
# and apply the patches needed to run them here.
#
# We ship patches rather than vendored copies so that what you reproduce is the
# published baseline plus a visible, reviewable delta -- not our snapshot of
# someone else's repository.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party patches

clone_at() {  # url dir commit
  if [ -d "third_party/$2/.git" ]; then
    echo "  third_party/$2 exists, skipping clone"
  else
    git clone --quiet "$1" "third_party/$2"
  fi
  git -C "third_party/$2" checkout --quiet "$3" 2>/dev/null \
    || echo "  ! could not check out $3 in $2; the patch may not apply"
}

apply_patch() {  # dir patchfile
  if [ ! -f "patches/$2" ]; then echo "  no patch for $1"; return; fi
  if git -C "third_party/$1" apply --check "../../patches/$2" 2>/dev/null; then
    git -C "third_party/$1" apply "../../patches/$2"
    echo "  applied patches/$2"
  else
    echo "  ! patches/$2 does not apply cleanly -- upstream has moved."
    echo "    The patch is small and commented; apply it by hand."
  fi
}

echo "Idea2Img (ECCV 2024)"
clone_at https://github.com/zyang-ur/Idea2Img.git idea2img HEAD
apply_patch idea2img idea2img_apiyi.patch
echo "  Four changes, each fixing something that made the baseline unrunnable"
echo "  here rather than something we disagreed with:"
echo "    - torch and cv2 imported lazily, so the API-only path needs no GPU stack"
echo "    - a t2i_apiyi generator, so the baseline runs on OUR generation model and"
echo "      the comparison isolates orchestration rather than backbone quality"
echo "    - message-format conversion: upstream builds 'content' as a list of bare"
echo "      strings, which the OpenAI-compatible endpoint rejects outright while"
echo "      the retry loop spins without ever surfacing the cause"
echo "    - token budget raised to 2048; at 512 a reasoning model spends the budget"
echo "      before emitting a full prompt and the parse fails on a truncated reply"

echo
echo "GenArtist (NeurIPS 2024) -- not run in the paper."
echo "  Its pipeline depends on component weights (BoxDiff, AnyDoor) we could not"
echo "  obtain in an offline evaluation environment. The clone is provided for"
echo "  completeness; the paper explains why we judged a second general-purpose"
echo "  generation agent unlikely to add information."
clone_at https://github.com/zhenyuw16/GenArtist.git GenArtist HEAD

echo
echo "done."
