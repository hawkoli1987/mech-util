#!/bin/bash
# Single script to launch PBS job and enter enroot container
#
# Usage: bash mech_client.sh [client_node] [server_node]
#   client_node: node number to run client on (default: 46)
#   server_node: node number where vLLM server runs (default: 45)

CLIENT_NODE="${1:-34}"
SERVER_NODE_LLM="${2:-34}"
SERVER_NODE_VLM="${3:-34}"
CLIENT_NAME="${4:-client_component}"

SETUP_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/mech_client_setup.sh"

echo "Launching PBS job on hopper-${CLIENT_NODE}"
echo "LLM server on hopper-${SERVER_NODE_LLM}"
echo "VLM server on hopper-${SERVER_NODE_VLM}"

qsub -I \
    -l select=1:mem=10gb:ngpus=1:ncpus=2:host=hopper-${CLIENT_NODE} \
    -l walltime=72:00:00 \
    -q AISG_debug \
    -N ${CLIENT_NAME} \
    -- /bin/bash "${SETUP_SCRIPT}" "hopper-${SERVER_NODE_LLM}" "hopper-${SERVER_NODE_VLM}" "${CLIENT_NAME}"
