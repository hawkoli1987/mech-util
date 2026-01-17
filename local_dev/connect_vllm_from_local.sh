#!/bin/bash
# Usage:
# bash mech-util/local_dev/connect_vllm_from_local.sh

# check if the port is already in use (do it in both local and hopper login node)
lsof -i :8001  # should be empty
lsof -i :8002  # check sibling ports

############################################################
# LLM server
############################################################
# binds hopper login node port 8001 to compute node 45 port 8001, 
ssh hopper \
  'ssh -f -N -L 8001:localhost:8001 hopper-45'

# binds local PC port 8001 to hopper login node port 8001, doing it in background
ssh -f -N -L 8001:localhost:8001 hopper


############################################################
# VLM server
############################################################
# binds hopper login node port 8002 to compute node 34 port 8002, 
ssh hopper \
  'ssh -f -N -L 8002:localhost:8002 hopper-34'

# binds local PC port 8002 to hopper login node port 8002, doing it in background
ssh -f -N -L 8002:localhost:8002 hopper


############################################################
# Embedding server
############################################################
# binds hopper login node port 8004 to compute node 34 port 8004, 
ssh hopper \
  'ssh -f -N -L 8004:localhost:8004 hopper-34'

# binds local PC port 8004 to hopper login node port 8004, doing it in background
ssh -f -N -L 8004:localhost:8004 hopper

############################################################
# test the port forwarding via curl
############################################################
echo "--- LLM ---" && \
# curl -s http://localhost:8001/v1/models
export OPENAI_API_BASE="http://localhost:8001"
curl -s ${OPENAI_API_BASE}/v1/models | \
python3 -c "import sys, json; print('\n'.join(m['id'] for m in json.load(sys.stdin)['data']))"

echo && echo "--- VLM ---" && \
# curl -s http://localhost:8002/v1/models
export OPENAI_API_BASE2="http://localhost:8002"
curl -s ${OPENAI_API_BASE2}/v1/models | \
python3 -c "import sys, json; print('\n'.join(m['id'] for m in json.load(sys.stdin)['data']))"

echo && echo "--- Embedding ---" && \
# curl -s http://localhost:8004/v1/models
export OPENAI_API_BASE3="http://localhost:8004"
curl -s ${OPENAI_API_BASE3}/v1/models | \
python3 -c "import sys, json; print('\n'.join(m['id'] for m in json.load(sys.stdin)['data']))"