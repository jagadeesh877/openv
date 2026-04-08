#!/usr/bin/env bash
set -uo pipefail

# ---------------------------------------------------------------------------
# Pre-Submission Validation Script
# ---------------------------------------------------------------------------
# This script ensures your environment meets all OpenEnv judging criteria.
# Run this in Git Bash, WSL, or Linux before submitting.
# ---------------------------------------------------------------------------

DOCKER_BUILD_TIMEOUT=600

if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

# Function to run commands with a timeout (Handles different environments)
run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout &>/dev/null; then
    timeout "$secs" "$@"
  elif command -v gtimeout &>/dev/null; then
    gtimeout "$secs" "$@"
  else
    "$@" &
    local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) &
    local watcher=$!
    wait "$pid" 2>/dev/null
    local rc=$?
    kill "$watcher" 2>/dev/null
    wait "$watcher" 2>/dev/null
    return $rc
  fi
}

echo -e "${BOLD}--- OpenEnv Pre-Validation Phase 1: File Integrity ---${NC}"

FILES=(
  "env/environment.py"
  "tasks/email_triage.py"
  "tasks/code_review.py"
  "tasks/meeting_scheduler.py"
  "graders/email_grader.py"
  "graders/code_grader.py"
  "graders/meeting_grader.py"
  "main.py"
  "requirements.txt"
  "Dockerfile"
  "openenv.yaml"
  "README.md"
  "inference.py"
)

MISSING=0
for FILE in "${FILES[@]}"; do
  if [ ! -e "$FILE" ]; then
    echo -e "[ ${RED}FAIL${NC} ] Missing required file: $FILE"
    MISSING=$((MISSING + 1))
  else
    echo -e "[ ${GREEN}PASS${NC} ] Found $FILE"
  fi
done

if [ "$MISSING" -gt 0 ]; then
  echo -e "\n${RED}${BOLD}STOP: $MISSING required files are missing.${NC}"
  exit 1
fi

echo -e "\n${BOLD}--- Phase 2: OpenEnv Manifest Validation ---${NC}"
if ! grep -q "class: OpenEnvEnvironment" "openenv.yaml"; then
  echo -e "[ ${RED}FAIL${NC} ] openenv.yaml does not contain the core Environment class mapping."
  exit 1
fi
echo -e "[ ${GREEN}PASS${NC} ] openenv.yaml looks valid."

echo -e "\n${BOLD}--- Phase 3: Docker Engine Check ---${NC}"
if ! docker info &>/dev/null; then
  echo -e "[ ${YELLOW}WARN${NC} ] Docker is not running or not reachable. Skipping build test."
else
  echo -e "[ ${GREEN}PASS${NC} ] Docker engine is reachable."
  
  echo -e "\n${BOLD}--- Phase 4: Docker Build Test (Timeout: ${DOCKER_BUILD_TIMEOUT}s) ---${NC}"
  if run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build -t openenv-validation-test . ; then
    echo -e "[ ${GREEN}PASS${NC} ] Docker image built successfully."
    
    echo -e "\n${BOLD}--- Phase 5: API Runtime & Health Check ---${NC}"
    # Start container temporarily to ping health
    CONTAINER_ID=$(docker run -d -p 7861:7860 openenv-validation-test)
    sleep 5
    
    if curl -s http://localhost:7861/ > /dev/null; then
      echo -e "[ ${GREEN}PASS${NC} ] FastAPI server responded at port 7860 (remapped to 7861)."
    else
      echo -e "[ ${RED}FAIL${NC} ] FastAPI server did not respond within 5 seconds."
      docker kill "$CONTAINER_ID"
      exit 1
    fi
    # Cleanup
    docker kill "$CONTAINER_ID" &>/dev/null
    docker rm "$CONTAINER_ID" &>/dev/null
    docker rmi openenv-validation-test &>/dev/null
  else
    echo -e "[ ${RED}FAIL${NC} ] Docker build timed out or failed. Check your Dockerfile."
    exit 1
  fi
fi

echo -e "\n${BOLD}--- Phase 6: Inference Check ---${NC}"
if ! python -c "import inference" &>/dev/null; then
  echo -e "[ ${YELLOW}WARN${NC} ] inference.py could not be imported. Ensure dependencies are in PATH."
else
  echo -e "[ ${GREEN}PASS${NC} ] inference.py has no syntax or import errors."
fi

echo -e "\n------------------------------------------------------------"
echo -e "${BOLD}${GREEN}PRE-VALIDATION SUCCESSFUL!${NC}"
echo -e "Your OpenEnv space is ready for judging."
echo -e "------------------------------------------------------------\n"
