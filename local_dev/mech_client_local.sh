#!/bin/bash
set -e

# Local macOS Docker workflow for component container inside tmux.
# Usage: bash mech-util/local_dev/mech_client_local.sh
# Env overrides: IMAGE_NAME, CONTAINER_NAME, PLATFORM
# Notes: if the tmux session already exists, this just attaches.

IMAGE_NAME="${IMAGE_NAME:-mech-component:local}"
CONTAINER_NAME="${CONTAINER_NAME:-mech_component_local}"
PLATFORM="${PLATFORM:-linux/amd64}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BUILD_SCRIPT="${SCRIPT_DIR}/docker_build_component.sh"

if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    echo "Docker image not found: ${IMAGE_NAME}"
    echo "Building with ${BUILD_SCRIPT}"
    bash "${BUILD_SCRIPT}"
fi

if [ -t 0 ]; then
    DOCKER_TTY="-it"
else
    DOCKER_TTY=""
fi

# Build docker --env args from common prefixes
env_args=""
prefixes=("NCCL" "CUDA" "SHARED" "HF" "WANDB" "XDG" "LOG" "CACHE" "TORCH" "TRITON" "VLLM" "OPENAI" "SERVER" "MECH")
for prefix in "${prefixes[@]}"; do
    while IFS= read -r var; do
        if [ -n "${!var}" ]; then
            env_args="${env_args} --env=${var}=${!var}"
        fi
    done < <(env | grep "^${prefix}_" | cut -d'=' -f1)
done

STARTUP_CMD="set -e; \
if ! pip show mech-util >/dev/null 2>&1; then \
  pip install -e /Mech/mech-util; \
fi; \
if ! pip show component-agent-cadquery >/dev/null 2>&1; then \
  pip install -e /Mech/component-agent-cadquery[test] --no-deps; \
fi; \
echo 'Container ready.'; \
exec /bin/bash"

TMP_SCRIPT="$(mktemp -t mech_docker_run.XXXXXX)"
cat > "${TMP_SCRIPT}" << EOF
#!/bin/bash
set -e
docker run --rm ${DOCKER_TTY} \\
    --name "${CONTAINER_NAME}" \\
    --platform "${PLATFORM}" \\
    --env MECH_CONTAINER_TYPE=freecad \\
    --env MECH_REPO_ROOT=/Mech \\
    ${env_args} \\
    -v "${REPO_ROOT}:/Mech" \\
    -w /Mech \\
    "${IMAGE_NAME}" \\
    /bin/bash -lc "${STARTUP_CMD}"
EOF
chmod +x "${TMP_SCRIPT}"

if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Attaching to running container: ${CONTAINER_NAME}"
    exec docker exec ${DOCKER_TTY} "${CONTAINER_NAME}" /bin/bash
fi

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Starting existing container: ${CONTAINER_NAME}"
    if [ -n "${DOCKER_TTY}" ]; then
        exec docker start -ai "${CONTAINER_NAME}"
    else
        exec docker start -a "${CONTAINER_NAME}"
    fi
fi

echo "Creating new container: ${CONTAINER_NAME}"
exec "${TMP_SCRIPT}"
