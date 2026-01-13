#!/bin/bash
set -e

# Accept server nodes as arguments (passed from mech_client.sh)
SERVER_NODE_LLM="${1:-}"
SERVER_NODE_VLM="${2:-}"
CLIENT_NAME="${3:-client_component}"

export SHARED_FS="/scratch_aisg/SPEC-SF-AISG"
export SHARED_FS2="/scratch/Projects/SPEC-SF-AISG"
SQSH_DIR="/scratch_aisg/SPEC-SF-AISG/sqsh"

if [ "${CLIENT_NAME}" = "client_component" ] || [ "${CLIENT_NAME}" = "client_assembly" ]; then
    SQSH_FILE="${SQSH_DIR}/freecad_py310_v1.sqsh"
    CONTAINER_NAME="freecad"
elif [ "${CLIENT_NAME}" = "client_simulation" ]; then
    SQSH_FILE="${SQSH_DIR}/calculix_py310_v1.sqsh"
    CONTAINER_NAME="calculix"
else
    SQSH_FILE="${SQSH_DIR}/vllm_mech_v2.sqsh"
    CONTAINER_NAME="vllm_mech"
fi

# Set enroot directories to avoid permission issues
# Ensure SHARED_FS is set and not empty
if [ -z "${SHARED_FS}" ]; then
    echo "ERROR: SHARED_FS is not set!"
    exit 1
fi

# Explicitly set and verify ENROOT paths (use double quotes for variable expansion)
export ENROOT_DATA_PATH="${SHARED_FS}/cache/.enroot/data"
export ENROOT_RUNTIME_PATH="${SHARED_FS}/cache/.enroot/runtime"

# Verify paths don't contain literal variable names (would indicate expansion failure)
if [[ "${ENROOT_DATA_PATH}" == *'${'* ]] || [[ "${ENROOT_DATA_PATH}" == *'$SHARED'* ]]; then
    echo "ERROR: ENROOT_DATA_PATH contains unexpanded variable: ${ENROOT_DATA_PATH}"
    echo "This usually happens when using single quotes instead of double quotes"
    exit 1
fi

mkdir -p "${ENROOT_DATA_PATH}" "${ENROOT_RUNTIME_PATH}"

# Ensure unsquashfs is in PATH (needed for enroot container creation)
# unsquashfs is typically in /usr/sbin which may not be in PATH in PBS jobs
export PATH="/usr/sbin:/usr/bin:/sbin:/bin:${PATH}"

# Verify unsquashfs is available
if ! command -v unsquashfs &> /dev/null; then
    echo "WARNING: unsquashfs not found in PATH. Trying to locate it..."
    if [ -f "/usr/sbin/unsquashfs" ]; then
        export PATH="/usr/sbin:${PATH}"
    else
        echo "ERROR: unsquashfs is required but not found. Please ensure squashfs-tools is installed."
        exit 1
    fi
fi

# Check if sqsh file exists
if [ ! -f "${SQSH_FILE}" ]; then
    echo "ERROR: SQSH file not found: ${SQSH_FILE}"
    exit 1
fi
echo "✓ SQSH file found: ${SQSH_FILE}"

# Create container if it doesn't exist
if ! enroot list 2>/dev/null | grep -q "^${CONTAINER_NAME}$"; then
    echo "Creating enroot container: ${CONTAINER_NAME} from ${SQSH_FILE}"
    enroot create -n "${CONTAINER_NAME}" "${SQSH_FILE}" || {
        echo "ERROR: Failed to create container ${CONTAINER_NAME}"
        exit 1
    }
    
    # Wait for container extraction to complete and verify /root exists
    CONTAINER_ROOT="${ENROOT_DATA_PATH}/${CONTAINER_NAME}/root"
    MAX_WAIT=30
    WAIT_COUNT=0
    while [ ! -d "${CONTAINER_ROOT}" ] && [ ${WAIT_COUNT} -lt ${MAX_WAIT} ]; do
        sleep 1
        WAIT_COUNT=$((WAIT_COUNT + 1))
    done
    
    # Ensure /root directory exists in the container (required for --root flag)
    if [ ! -d "${CONTAINER_ROOT}" ]; then
        echo "WARNING: /root directory not found after container creation, creating it..."
        mkdir -p "${CONTAINER_ROOT}"
        # Set proper permissions for root directory (700 = rwx------)
        chmod 700 "${CONTAINER_ROOT}" 2>/dev/null || true
    else
        echo "✓ Container /root directory verified"
    fi
    
    # Verify container is properly listed
    if ! enroot list | grep -q "${CONTAINER_NAME}"; then
        echo "ERROR: Container ${CONTAINER_NAME} was not created successfully"
        exit 1
    fi
fi

# Server connection - use arguments if provided, otherwise fall back to defaults
export SERVER_NODE_LLM="${SERVER_NODE_LLM:-hopper-46}"
export SERVER_NODE_VLM="${SERVER_NODE_VLM:-hopper-34}"
export OPENAI_API_BASE="http://${SERVER_NODE_LLM}:8001"
export OPENAI_API_BASE2="http://${SERVER_NODE_VLM}:8002"

# Export container name for startup script
export MECH_CONTAINER_TYPE="${CONTAINER_NAME}"

# Build enroot --env arguments
env_args=""
prefixes=("NCCL" "CUDA" "SHARED" "HF" "WANDB" "XDG" "LOG" "CACHE" "TORCH" "TRITON" "VLLM" "OPENAI" "SERVER" "MECH")

for prefix in "${prefixes[@]}"; do
    while IFS= read -r var; do
        if [ -n "${!var}" ]; then
            env_args="${env_args} --env=${var}=${!var}"
        fi
    done < <(env | grep "^${prefix}_" | cut -d'=' -f1)
done

# Create rc script to bypass container entrypoint
RC_SCRIPT="/tmp/enroot_rc_$$.sh"
cat > "${RC_SCRIPT}" << 'RCEOF'
#!/bin/bash
exec "$@"
RCEOF
chmod +x "${RC_SCRIPT}"

# Create startup script in HOME (accessible inside container via mount)
STARTUP_SCRIPT="${HOME}/.container_startup_$$.sh"
cat > "${STARTUP_SCRIPT}" << 'STARTUPEOF'
#!/bin/bash
echo "=================================================="
echo "Container type: ${MECH_CONTAINER_TYPE}"
echo "Installing mech packages in editable mode..."
echo "=================================================="

# Check if mech-util is already installed in editable mode
# Editable installs reflect source changes immediately - only need to install once per container session
if ! pip show mech-util >/dev/null 2>&1; then
    echo "Installing mech-util..."
    pip install -e /scratch/Projects/SPEC-SF-AISG/source_files/Mech/mech-util
else
    echo "mech-util already installed (editable mode reflects changes immediately)"
fi

# Install container-specific packages
if [ "${MECH_CONTAINER_TYPE}" = "calculix" ]; then
    if ! pip show simulation-agent-calculix >/dev/null 2>&1; then
        echo "Installing simulation-agent-calculix..."
        pip install -e /scratch/Projects/SPEC-SF-AISG/source_files/Mech/simulation-agent-calculix[test]
    else
        echo "simulation-agent-calculix already installed (editable mode reflects changes immediately)"
    fi
elif [ "${MECH_CONTAINER_TYPE}" = "freecad" ]; then
    if ! pip show component-agent-cadquery >/dev/null 2>&1; then
        echo "Installing component-agent-cadquery..."
        pip install -e /scratch/Projects/SPEC-SF-AISG/source_files/Mech/component-agent-cadquery[test]
    else
        echo "component-agent-cadquery already installed (editable mode reflects changes immediately)"
    fi
fi

echo "=================================================="
echo "Checking tools for ${MECH_CONTAINER_TYPE} container..."
echo "=================================================="

# Check container-specific tools
if [ "${MECH_CONTAINER_TYPE}" = "calculix" ]; then
    # Simulation container: check for CalculiX, Gmsh, CGX
    if command -v ccx &> /dev/null; then
        echo "✓ CalculiX (ccx) found: $(which ccx)"
    else
        echo "✗ CalculiX (ccx) not found"
    fi
    if command -v gmsh &> /dev/null; then
        echo "✓ Gmsh found: $(which gmsh)"
    else
        echo "✗ Gmsh not found"
    fi
    if command -v cgx &> /dev/null; then
        echo "✓ CGX found: $(which cgx)"
    else
        echo "✗ CGX not found"
    fi
elif [ "${MECH_CONTAINER_TYPE}" = "freecad" ]; then
    # Component/Assembly container: check for FreeCAD, CadQuery
    if python3 -c "import FreeCAD" 2>/dev/null; then
        echo "✓ FreeCAD Python module available"
    else
        echo "✗ FreeCAD Python module not available"
    fi
    if python3 -c "import cadquery" 2>/dev/null; then
        echo "✓ CadQuery available"
    else
        echo "✗ CadQuery not available"
    fi
    if command -v gmsh &> /dev/null; then
        echo "✓ Gmsh found: $(which gmsh)"
    else
        echo "✗ Gmsh not found (optional for meshing)"
    fi
else
    # vLLM container: minimal checks
    echo "vLLM container - no specific tools required"
fi

echo "=================================================="
echo "Checking vLLM servers..."
echo "=================================================="

# Check LLM server
LLM_MODEL=$(curl -s ${OPENAI_API_BASE}/v1/models 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'] if d.get('data') else '')" 2>/dev/null)
if [ -n "$LLM_MODEL" ]; then
    echo "✓ LLM Server (${OPENAI_API_BASE}): ${LLM_MODEL}"
else
    echo "✗ LLM Server (${OPENAI_API_BASE}): Not available"
fi

# Check VLM server
VLM_MODEL=$(curl -s ${OPENAI_API_BASE2}/v1/models 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'] if d.get('data') else '')" 2>/dev/null)
if [ -n "$VLM_MODEL" ]; then
    echo "✓ VLM Server (${OPENAI_API_BASE2}): ${VLM_MODEL}"
else
    echo "✗ VLM Server (${OPENAI_API_BASE2}): Not available"
fi

echo "=================================================="
echo ""

# Start interactive bash
exec /bin/bash
STARTUPEOF
chmod +x "${STARTUP_SCRIPT}"

echo "=================================================="
echo "Starting enroot container: ${CONTAINER_NAME}"
echo "LLM server: ${SERVER_NODE_LLM}:8001"
echo "VLM server: ${SERVER_NODE_VLM}:8002"
echo "=================================================="

# Start container with startup script
exec enroot start \
    --root \
    --rw \
    --rc "${RC_SCRIPT}" \
    --mount "${HOME}:${HOME}" \
    --mount "${SHARED_FS}:${SHARED_FS}" \
    --mount "${SHARED_FS2}:${SHARED_FS2}" \
    ${env_args} \
    ${CONTAINER_NAME} \
    /bin/bash "${STARTUP_SCRIPT}"
