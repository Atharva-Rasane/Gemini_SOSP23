FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive

ENV CUDA_HOME=/usr/local/cuda \
    DEEPSPEED_DISABLE_MPI=1 \
    GEMINI_CHECKPOINT_DIR=/checkpoints/gemini \
    HF_HOME=/models/huggingface \
    TRANSFORMERS_CACHE=/models/huggingface/transformers \
    MAX_JOBS=2 \
    NCCL_DEBUG=WARN \
    PDSH_RCMD_TYPE=ssh \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libaio-dev \
        ninja-build \
        openssh-client \
        openssh-server \
        pdsh \
        rsync \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd /models/huggingface /checkpoints/gemini /workspace

WORKDIR /workspace/gemini

COPY requirements/requirements.txt requirements/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir -r requirements/requirements.txt \
    && python -m pip install --no-cache-dir "transformers==4.24.0" boto3 \
    && rm -rf /root/.cache/pip

COPY . .

RUN TORCH_CUDA_ARCH_LIST=7.5 DS_BUILD_OPS=0 python -m pip install --no-cache-dir -e . \
    && python -c 'import torch, deepspeed, transformers; assert torch.__version__.startswith("1.13.0"); assert deepspeed.__version__.startswith("0.7.3"); assert transformers.__version__ == "4.24.0"; print(torch.__version__, deepspeed.__version__, transformers.__version__)' \
    && rm -rf /root/.cache/pip

COPY docker/entrypoint.sh /usr/local/bin/gemini-entrypoint
RUN chmod 0755 /usr/local/bin/gemini-entrypoint \
    && printf '%s\n' \
        'PermitRootLogin prohibit-password' \
        'PasswordAuthentication no' \
        'PubkeyAuthentication yes' \
        >> /etc/ssh/sshd_config \
    && passwd -d root \
    && rm -f /etc/ssh/ssh_host_*

EXPOSE 2223 2379 2380 29500-29600
ENTRYPOINT ["/usr/local/bin/gemini-entrypoint"]
CMD ["bash"]
