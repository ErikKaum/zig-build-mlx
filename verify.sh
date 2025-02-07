#!/usr/bin/env bash
# adapted from https://github.com/mitchellh/zig-build-libxml2/blob/main/verify.sh
#
# Verify the contents of the upstream/<repo>. This compares to the "upstream_<repo>.txt"
# file for the upstream ref.
set -e

if [ $# -eq 0 ]; then
    echo "Error: Repository name argument is required"
    echo "Usage: ./verify.sh <repo>"
    echo "  repo: mlx, doctest, or fmt"
    exit 1
fi

repo=${1:-mlx}
if [[ ! "$repo" =~ ^(mlx|doctest|fmt)$ ]]; then
    echo "Error: Repository must be either 'mlx', 'doctest', or 'fmt'"
    exit 1
fi

# Checksum a directory
function checksum {
    # First clear all extended attributes
    find "$1" -exec xattr -c {} + 2>/dev/null || true
    
    # Create tar archive with consistent ordering and no metadata
    (cd "$1" && \
     find . -type f -print0 | \
     LC_ALL=C sort -z | \
     xargs -0 cat \
    ) | shasum -a256
}

ref=$(cat upstream_${repo}.txt | tr -d "[:space:]")
tmp=$(mktemp -d)

./update.sh $repo $ref $tmp >/dev/null 2>&1

actual=$(checksum upstream/$repo)
expected=$(checksum $tmp/$repo)
if [ "$actual" != "$expected" ]; then
    echo "Upstream verification failed for $repo!"
    echo "Expected: $expected"
    echo "Actual:   $actual"
    exit 1
fi

echo "Upstream verification successful for $repo"
echo "Expected: $expected"
echo "Actual:   $actual"
exit 0