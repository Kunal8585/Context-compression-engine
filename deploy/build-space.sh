#!/usr/bin/env bash
# Assemble the Hugging Face Space repo from this project.
#
#   ./deploy/build-space.sh            -> builds deploy/.space-build/
#   ./deploy/build-space.sh --verify   -> also builds the image locally and smoke-tests it
#
# The Space is its own git repo, so this copies only what the hosted backend
# needs. Push instructions are printed at the end.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/deploy/.space-build"
cd "$ROOT"

info() { printf '\033[1;34m==>\033[0m %s\n' "$1"; }
fail() { printf '\033[1;31m[x]\033[0m %s\n' "$1"; exit 1; }

[ -f reports/latest.json ] || fail \
  "reports/latest.json is missing - run 'python -m eval.harness' first, or the
   deployed dashboard will have no metrics to show."

info "Assembling Space repo in deploy/.space-build"
rm -rf "$OUT"
mkdir -p "$OUT/deploy/hf-space"

cp deploy/hf-space/Dockerfile        "$OUT/deploy/hf-space/"
cp deploy/hf-space/requirements.txt  "$OUT/deploy/hf-space/"
cp deploy/hf-space/README.md         "$OUT/README.md"   # Space card must be at root
cp config.yaml                       "$OUT/"

for dir in engine backend eval reports; do
  rsync -a --exclude='__pycache__' --exclude='*.pyc' "$dir/" "$OUT/$dir/"
done
# Sample corpus powers the /samples endpoints; the test set powers /evaluate.
mkdir -p "$OUT/data"
rsync -a --exclude='__pycache__' data/sample_corpus/ "$OUT/data/sample_corpus/"
rsync -a data/eval/ "$OUT/data/eval/"

# HF Spaces expects the Dockerfile at the repo root.
cp deploy/hf-space/Dockerfile "$OUT/Dockerfile"
# ...but the Dockerfile COPYs from deploy/hf-space/requirements.txt, so keep the
# path it references valid inside the build context too.

printf '__pycache__/\n*.pyc\n.venv/\n' > "$OUT/.gitignore"

info "Space repo ready: $(du -sh "$OUT" | cut -f1) in $OUT"

if [ "${1:-}" = "--verify" ]; then
  command -v docker >/dev/null 2>&1 || fail "docker not installed; skip --verify"
  info "Building the image locally (this takes a few minutes)"
  docker build -t cce-space "$OUT"
  info "Starting it on :7861"
  docker rm -f cce-space-test >/dev/null 2>&1 || true
  docker run -d --name cce-space-test -p 7861:7860 cce-space >/dev/null
  for _ in $(seq 1 90); do
    curl -sf http://localhost:7861/health >/dev/null 2>&1 && break
    sleep 2
  done
  curl -sf http://localhost:7861/health >/dev/null 2>&1 \
    || { docker logs cce-space-test | tail -30; fail "container did not become healthy"; }
  info "Health OK. Smoke-testing /compress and /evaluate"
  curl -s -X POST http://localhost:7861/compress \
    -H 'Content-Type: application/json' \
    -d '{"text":"def a(x):\n    return x+1\n\ndef b(y):\n    return y+1\n","name":"t.py","budget_ratio":0.5}' \
    | head -c 300
  echo
  curl -s -X POST http://localhost:7861/evaluate -H 'Content-Type: application/json' -d '{"run":false}' | head -c 200
  echo
  docker rm -f cce-space-test >/dev/null
  info "Image verified"
fi

cat <<EOF

Next steps (these need your Hugging Face account - I cannot authenticate for you):

  1. Create a Space:  https://huggingface.co/new-space
       SDK: Docker      Hardware: CPU basic (free)

  2. Push:
       cd $OUT
       git init && git add -A && git commit -m "Context compression engine API"
       git remote add origin https://huggingface.co/spaces/<user>/<space>
       git push -u origin main

     (If prompted, authenticate with:  huggingface-cli login)

  3. First build takes ~10 min - it bakes MiniLM and spaCy into the image.
     Watch the Build logs tab.

  4. Then point the frontend at it:
       cd $ROOT/frontend
       echo "VITE_API_URL=https://<user>-<space>.hf.space" > .env.production
       npm run build
EOF
