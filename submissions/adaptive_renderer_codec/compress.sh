#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# usage: compress.sh <PR #146 (joint_hops) archive.zip>
# Re-codes that archive's renderer weights losslessly; refuses unless every gate passes.
python "$HERE/tools/check_lossless.py" --reference "$1" --out "$HERE/archive.zip"
