#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SCRIPT_DIR}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PATH="${HOME}/.local/bin:${PATH}"

HOSTFILE="${HOSTFILE:-${SCRIPT_DIR}/../hostfile}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-pretrain_gpt.py}"
TRAIN_CONFIG="${TRAIN_CONFIG:-tiny_gpt_template.json}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-29500}"

JOB_NAME="${JOB_NAME:-tiny_gpt}"
OUTPUT_DIR="${OUTPUT_DIR:-${GEMINI_CHECKPOINT_DIR:-${SCRIPT_DIR}/${JOB_NAME}}/output}"
SNAPSHOT_PATH="${SNAPSHOT_PATH:-${GEMINI_CHECKPOINT_DIR:-${SCRIPT_DIR}/${JOB_NAME}}/snapshot}"
MAX_STEPS="${MAX_STEPS:-30}"
PRINT_STEPS="${PRINT_STEPS:-1}"
COMM_PROFILE_STEPS="${COMM_PROFILE_STEPS:-3}"
JUMP_PROFILE_LINES="${JUMP_PROFILE_LINES:-1}"
SNAPSHOT_MODE="${SNAPSHOT_MODE:-interleave}"
NETWORK_BANDWIDTH="${NETWORK_BANDWIDTH:-80}"
SNAPSHOT_BUFFER_SIZE="${SNAPSHOT_BUFFER_SIZE:-1}"
SPAN_THRESHOLD="${SPAN_THRESHOLD:-100}"
SPAN_ALPHA="${SPAN_ALPHA:-0.8}"
MAX_BLOCKS_IN_SPAN="${MAX_BLOCKS_IN_SPAN:-1}"
LOG_FILE="${LOG_FILE:-${GEMINI_CHECKPOINT_DIR:-${SCRIPT_DIR}/${JOB_NAME}}/log_${JOB_NAME}}"

mkdir -p "${OUTPUT_DIR}" "${SNAPSHOT_PATH}" "$(dirname "${LOG_FILE}")"

if [ -n "${DEEPSPEED_BIN:-}" ]; then
    deepspeed="${DEEPSPEED_BIN}"
elif command -v deepspeed >/dev/null 2>&1; then
    deepspeed="$(command -v deepspeed)"
elif command -v ds >/dev/null 2>&1; then
    deepspeed="$(command -v ds)"
elif python3 -c "import deepspeed" >/dev/null 2>&1; then
    deepspeed="python3 -m deepspeed.launcher.runner"
else
    echo "DeepSpeed is not installed or importable. Run: python3 -m pip install -e ${REPO_ROOT}" >&2
    exit 1
fi

if [ ! -f "${HOSTFILE}" ]; then
    echo "Hostfile not found: ${HOSTFILE}" >&2
    echo "Create ${SCRIPT_DIR}/../hostfile or set HOSTFILE=/path/to/hostfile." >&2
    exit 1
fi

ds_cmd="\
${deepspeed} --hostfile=${HOSTFILE} --master_port=${MASTER_PORT} ${MASTER_ADDR:+--master_addr=${MASTER_ADDR}} ${TRAIN_SCRIPT} \
--deepspeed \
--deepspeed_config $TRAIN_CONFIG \
--job_name ${JOB_NAME} \
--max_steps ${MAX_STEPS} \
--print_steps ${PRINT_STEPS} \
--output_dir ${OUTPUT_DIR} \
--snapshot_path ${SNAPSHOT_PATH} \
--comm_profile_steps ${COMM_PROFILE_STEPS} \
--jump_profile_lines ${JUMP_PROFILE_LINES} \
--enable_comm_profile \
--snapshot_mode ${SNAPSHOT_MODE} \
--network_bandwidth ${NETWORK_BANDWIDTH} \
--snapshot_buffer_size ${SNAPSHOT_BUFFER_SIZE} \
--span_threshold ${SPAN_THRESHOLD} \
--span_alpha ${SPAN_ALPHA} \
--max_blocks_in_span ${MAX_BLOCKS_IN_SPAN} \
--save_to_disk \
# --enable_snapshot_profile \
"

echo $ds_cmd
eval $ds_cmd 2>&1 | tee ${LOG_FILE}
