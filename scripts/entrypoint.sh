#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  set -- bash
fi
exec "$@"
