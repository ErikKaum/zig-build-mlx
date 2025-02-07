#!/usr/bin/env bash
# adapted from https://github.com/mitchellh/zig-build-libxml2/blob/main/update.sh
#
# Update the upstream to a specific commit. If this fails then it may leave
# your working tree in a bad state. You can recover by using Git to reset:
# git reset --hard HEAD.
set -e

if [ $# -eq 0 ]; then
    echo "Error: Repository name argument is required"
    echo "Usage: ./update.sh <repo> [ref] [out]"
    echo "  repo: mlx, doctest, or fmt"
    echo "  ref:  git reference (default: HEAD)"
    echo "  out:  output directory (default: upstream)"
    exit 1
fi

repo=${1:-mlx}
ref=${2:-HEAD}
out=${3:-upstream}

mkdir -p $out/$repo

if [ "$repo" = "mlx" ]; then
    git clone https://github.com/ml-explore/mlx.git $out/$repo
    git -C $out/$repo checkout $ref
    git -C $out/$repo rev-parse HEAD > ${out}_${repo}.txt
elif [ "$repo" = "doctest" ]; then
    git clone https://github.com/doctest/doctest.git $out/$repo
    git -C $out/$repo checkout $ref
    git -C $out/$repo rev-parse HEAD > ${out}_${repo}.txt
elif [ "$repo" = "fmt" ]; then
    git clone https://github.com/fmtlib/fmt.git $out/$repo
    git -C $out/$repo checkout $ref
    git -C $out/$repo rev-parse HEAD > ${out}_${repo}.txt
else
    echo "Error: First argument must be either 'mlx', 'doctest', or 'fmt'"
    exit 1
fi

rm -rf $out/$repo/.git

