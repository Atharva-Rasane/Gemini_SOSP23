#!/usr/bin/env bash
set -euo pipefail

SSH_PORT="${SSH_PORT:-2223}"
export SSH_PORT

if [ -z "${PDSH_SSH_ARGS_APPEND:-}" ]; then
    PDSH_SSH_ARGS_APPEND="-p ${SSH_PORT} -o StrictHostKeyChecking=no"
fi
export PDSH_SSH_ARGS_APPEND

if [ -z "${NCCL_SOCKET_IFNAME:-}" ]; then
    NCCL_SOCKET_IFNAME="$(awk '$2 == "00000000" { print $1; exit }' /proc/net/route)"
fi
if [ -z "$NCCL_SOCKET_IFNAME" ]; then
    echo "Unable to detect the default network interface; set NCCL_SOCKET_IFNAME." >&2
    exit 1
fi
export NCCL_SOCKET_IFNAME

mkdir -p "${HF_HOME:-/models/huggingface}" "${GEMINI_CHECKPOINT_DIR:-/checkpoints/gemini}"

if [ "${START_SSHD:-0}" = "1" ]; then
    install -d -m 0700 /root/.ssh
    if [ -d /ssh-host ]; then
        cp -a /ssh-host/. /root/.ssh/
        chown -R root:root /root/.ssh
        chmod 0700 /root/.ssh
        find /root/.ssh -type f -exec chmod 0600 {} +
        find /root/.ssh -type f -name '*.pub' -exec chmod 0644 {} +
    fi
    cat > /root/.ssh/config <<EOF
Host *
    Port ${SSH_PORT}
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
EOF
    chown root:root /root/.ssh/config
    chmod 0600 /root/.ssh/config
    ssh-keygen -A
    mkdir -p /run/sshd
    /usr/sbin/sshd -p "$SSH_PORT"
fi

exec "$@"
