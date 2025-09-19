#!/usr/bin/env bash

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[1;34m'
YELLOW='\033[1;33m'
RESET='\033[0m'

cd "$(dirname "$0")"

if ! command -v cargo-sort &> /dev/null; then
    cargo install cargo-sort --locked
fi

dir=$(basename "$(pwd)")

for command in \
    "cargo clippy --fix --allow-dirty --allow-staged" \
    "cargo +nightly fmt" \
    "cargo sort --workspace > /dev/null" \
    "cargo test" \
; do
    if eval "$command"; then
        echo -e "${GREEN}✔${RESET} ${BLUE}[$dir]${RESET} $command"
    else
        exit_status=$?
        echo -e "${RED}✘${RESET} ${BLUE}[$dir]${RESET} $command failed with exit code ${exit_status}" >&2
        exit $exit_status
    fi
done
