#!/usr/bin/env bash
# Build the wasm bundle with threads (wasm-bindgen-rayon) into ./pkg.
#
# build-std is passed here (not in .cargo/config.toml) so it only applies to the wasm build,
# keeping host `cargo test` from rebuilding std. The atomics/shared-memory rustflags live in
# .cargo/config.toml (scoped to the wasm32 target).
set -euo pipefail

PROFILE="${1:-release}"   # release | dev
FLAG="--release"
[ "$PROFILE" = "dev" ] && FLAG="--dev"

echo ">> wasm-pack build ($PROFILE) with build-std + atomics"
# wasm-pack's CLI is `build [OPTIONS] [PATH] [EXTRA_OPTIONS]...`. We must pass an explicit
# PATH (`.`) before `--`, otherwise the first cargo flag (`-Z`) is mistaken for PATH.
# Default out-dir is ./pkg.
wasm-pack build . --target web "$FLAG" \
  -- -Z build-std=panic_abort,std

# wasm-bindgen-rayon 1.3.0 emits workerHelpers.js into snippets/<hash>/src/ (one level
# deeper than its `import('../../..')` assumes), so without a bundler that import resolves to
# the pkg/ DIRECTORY instead of the glue module. Repoint it at the actual module file.
HELPER=$(find pkg/snippets -name workerHelpers.js | head -1)
if [ -n "$HELPER" ] && grep -q "import('../../..')" "$HELPER"; then
  sed -i "s#import('../../..')#import('../../../ecology_simulator.js')#" "$HELPER"
  echo ">> patched rayon workerHelpers.js module import -> ../../../ecology_simulator.js"
fi

echo ">> done. Run:  python3 serve.py   then open http://localhost:8080/"
