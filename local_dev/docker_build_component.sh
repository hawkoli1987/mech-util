#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-mech-component:local}"
PLATFORM="${PLATFORM:-linux/amd64}"

echo "Building Docker image: ${IMAGE_NAME} (${PLATFORM})"
docker build \
    --platform "${PLATFORM}" \
    -t "${IMAGE_NAME}" \
    -f "${SCRIPT_DIR}/Dockerfile.component" \
    "${SCRIPT_DIR}"
