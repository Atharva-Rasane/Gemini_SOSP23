# Gemini SOSP'23 Docker image and two-node GCP runbook

This image contains the Gemini/DeepSpeed source code and its pinned runtime, but
does not contain model weights, tokenizer caches, or generated checkpoints.

The tested split is:

- image code and dependencies: `/workspace/gemini`
- model/tokenizer cache: `/models/huggingface`
- Gemini snapshots, logs, and DeepSpeed outputs: `/checkpoints/gemini`

The image uses the artifact's documented software generation: CUDA 11.6,
PyTorch 1.13.0, this repository's DeepSpeed 0.7.3 fork, and Transformers
4.24.0. The Docker build targets Tesla T4 compute capability 7.5.

## What the two-node smoke test runs

`examples/GPT/launch.sh` launches one ZeRO-3 worker per VM. The tiny GPT model
is randomly initialized from `tiny_gpt_template.json`; no pretrained model
weights are downloaded. Transformers downloads only the GPT-2 tokenizer into
the mounted model cache.

Gemini first profiles all-gather and reduce-scatter gaps. In interleave mode it
then divides optimizer state into blocks and sends blocks to the peer rank.
When the run exits, each node stores its local optimizer state and its peer
replica beneath `/checkpoints/gemini/snapshot`.

The AWS Auto Scaling/etcd controller under
`deepspeed/runtime/snapshot/launch.py` is a separate failure-recovery path.
The basic two-node training and snapshot test does not require etcd.

## Build

Run from the repository root:

```bash
docker build -f docker/Dockerfile -t gemini-sosp23:cuda11.6 .
docker run --rm --gpus all gemini-sosp23:cuda11.6 \
  python -c 'import torch, deepspeed, transformers; print(torch.cuda.get_device_name(0), deepspeed.__version__, transformers.__version__)'
```

The build context must be the repository root, not `docker/`.

## Push to the existing Artifact Registry repository

```bash
export PROJECT_ID=gbc-oit-rc-basil-app-bo
export REGION=us-central1
export REPOSITORY=cloud-run-source-deploy
export IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/gemini-sosp23:cuda11.6"

gcloud auth configure-docker "${REGION}-docker.pkg.dev"
docker tag gemini-sosp23:cuda11.6 "${IMAGE}"
docker push "${IMAGE}"
```

Cloud Build is also supported:

```bash
gcloud builds submit . --project "${PROJECT_ID}" \
  --config docker/cloudbuild.yaml \
  --substitutions "_IMAGE=${IMAGE}"
```

## Tested VMs

Both VMs are in `us-central1-a`, use the network tag
`pccheck-distributed`, and have one Tesla T4:

- `atharva-pccheck-t4-1` — `10.128.0.60`
- `atharva-pccheck-t4-2` — `10.128.0.59`

The tag-to-tag firewall rule must allow TCP, UDP, and ICMP between the two
instances. NCCL may select dynamic TCP ports; limiting the rule to only the
rendezvous port is insufficient. Keep the source and target restricted to the
cluster network tag.

The hosts already keep the dedicated cluster SSH material under
`$HOME/distributed-ssh`. It is mounted read-only and copied into each
container at startup.

## Start one container on each VM

Gemini defaults to container SSH port 2223, so it can coexist with the PCcheck
container on port 2222. It auto-detects the default network interface; do not
hardcode `ens4` or `ens5`.

```bash
export IMAGE=us-central1-docker.pkg.dev/gbc-oit-rc-basil-app-bo/cloud-run-source-deploy/gemini-sosp23:cuda11.6

gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
docker pull "${IMAGE}"
docker run -d --name gemini --restart unless-stopped \
  --gpus all --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e START_SSHD=1 \
  -e SSH_PORT=2223 \
  -e MASTER_ADDR=10.128.0.60 \
  -v "$HOME/distributed-ssh:/ssh-host:ro" \
  -v gemini-models:/models \
  -v gemini-checkpoints:/checkpoints \
  "${IMAGE}" sleep infinity
```

The entrypoint configures raw OpenSSH and pdsh to use port 2223. This matters
because DeepSpeed 0.7.3 does not provide a `--ssh_port` launcher option.

Create the same hostfile inside both containers:

```bash
docker exec gemini bash -c \
  'printf "%s\n" "root@10.128.0.60 slots=1" "root@10.128.0.59 slots=1" > /workspace/gemini/examples/hostfile'
```

Verify GPU access, mounts, and container-to-container SSH:

```bash
docker exec gemini nvidia-smi
docker exec gemini ssh root@10.128.0.60 hostname
docker exec gemini ssh root@10.128.0.59 hostname
docker exec gemini sh -c 'test -w /models && test -w /checkpoints'
```

## Run distributed Gemini training

Start the run from VM 1:

```bash
docker exec -it gemini bash -lc '
  cd /workspace/gemini &&
  MASTER_ADDR=10.128.0.60 \
  HOSTFILE=/workspace/gemini/examples/hostfile \
  TRAIN_CONFIG=tiny_gpt_template.json \
  MAX_STEPS=8 \
  COMM_PROFILE_STEPS=3 \
  bash examples/GPT/launch.sh
'
```

Use at least one step after `COMM_PROFILE_STEPS` so interleave mode can compute
and exercise its snapshot strategy. The first run downloads the GPT-2 tokenizer
to `/models/huggingface` and JIT-builds any required DeepSpeed CUDA extension,
so it is slower than subsequent runs.

Inspect persistent outputs on each node:

```bash
docker exec gemini find /checkpoints/gemini -maxdepth 3 -type f -printf '%p %s bytes\n'
```

Expected snapshot filenames for two ranks are:

- rank 0 node: `0.pt`, `0_1.pt`, and `0.json`
- rank 1 node: `1.pt`, `1_0.pt`, and `1.json`

The named volumes survive container replacement. Removing the volumes deletes
the cached tokenizer or checkpoints, so do that only when those artifacts are
no longer needed.

## Operational notes

- The paper's multi-billion parameter configurations require many more GPUs and
  much more memory than two T4s. Start with `tiny_gpt_template.json`.
- Use `--network host`; DeepSpeed, NCCL, and peer snapshot traffic must reach
  the other VM's container.
- Do not expose SSH port 2223 or etcd publicly.
- The two GPU VMs remain billable while running. Stop them when the experiment
  is idle.
