#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${SCRIPT_DIR}/.cargo"

config_file="${SCRIPT_DIR}/.cargo/config.toml"
patch_url='https://github.com/mickvangelderen/micrograd-rs.git'

if [[ -f "$config_file" ]] && grep -q "^\[patch\.\"$patch_url\"\]" "$config_file"; then
    echo -e "\033[31merror\033[0m: Patch for $patch_url already exists!"
    echo "Remove the patch from $config_file manually and run this script again."
    exit 1
fi

cat >> "$config_file" << EOF
[patch."$patch_url"]
micrograd-rs = { path = "../micrograd-rs" }
EOF
