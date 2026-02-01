#!/bin/bash
set -e

REPO="gyudonlol/agentbeats-rlm"
PLATFORMS="linux/amd64,linux/arm64"

# Ensure buildx builder exists and supports multi-platform
docker buildx create --name multiarch --use 2>/dev/null || docker buildx use multiarch

echo "Building and pushing green agent..."
docker buildx build \
  --platform "$PLATFORMS" \
  -f Dockerfile \
  -t "$REPO:latest" \
  --push \
  .

echo "Building and pushing purple (white) agent..."
docker buildx build \
  --platform "$PLATFORMS" \
  -f Dockerfile.purple \
  -t "$REPO-purple:latest" \
  --push \
  .

echo "Done! Images pushed:"
echo "  - $REPO:latest"
echo "  - $REPO-purple:latest"
