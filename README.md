[![License Apache 2.0](https://badgen.net/badge/license/apache2.0/blue)](https://github.com/Microsoft/DeepSpeed/blob/master/LICENSE) 


# Artifact Evaluation of SOSP 2023 #30

This repository contains the system code and scripts that help run the experiments of our SOSP '23 paper (#30 in AE).


## Prerequisites

Minimum tested versions:

- DeepSpeed == 0.7.3
- CUDA == 11.6
- PyTorch == 1.13.0+cu116
- NCCL == 2.14.3
- etcd == 3.5.0
- Auto Scaling Group in AWS
- Git, Python 3, pip, OpenSSH, Docker, and build tools
- OS: Linux and other OS supported by DeepSpeed

### Install prerequisites on a blank Ubuntu/Debian VM

Run this on each VM before cloning or running the artifact. The VM must be a GPU VM with NVIDIA drivers installed; `nvidia-smi` must show at least one GPU.

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  git \
  python3-dev \
  python3-pip \
  python3-venv \
  openssh-client \
  openssh-server \
  docker.io \
  pdsh

sudo systemctl enable --now ssh
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
```

Log out and back in after adding the user to the `docker` group, or keep using `sudo docker` in the current shell.

Verify GPU and driver visibility:

```bash
nvidia-smi
```

Install the minimum Python package versions used by this artifact:

```bash
python3 -m pip install --upgrade pip
python3 -m pip uninstall -y torch torchvision torchaudio numpy pydantic deepspeed
python3 -m pip install \
  torch==1.13.0+cu116 \
  torchvision==0.14.0+cu116 \
  torchaudio==0.13.0 \
  --extra-index-url https://download.pytorch.org/whl/cu116
python3 -m pip install numpy==1.23.5 pydantic==1.10.13 matplotlib==3.7.5 transformers==4.24.0 boto3
```

After cloning this repository, install the artifact itself:

```bash
python3 -m pip install -r requirements/requirements.txt
python3 -m pip install -e .
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD:$PYTHONPATH"
```

Verify the installed versions:

```bash
python3 - <<'PY'
import torch, deepspeed, numpy, pydantic
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("gpu count", torch.cuda.device_count())
print("nccl", torch.cuda.nccl.version())
print("deepspeed", deepspeed.__version__)
print("numpy", numpy.__version__)
print("pydantic", pydantic.__version__)
PY
```

Start etcd 3.5 with Docker on the VM that will host etcd:

```bash
export VM1_PRIVATE_IP=$(hostname -I | awk '{print $1}')
export ETCD_VERSION=v3.5.0

sudo docker rm -f etcd_container || true
sudo docker volume create --name etcd-data
sudo docker run --rm -d \
  -p 2379:2379 -p 2380:2380 \
  -v etcd-data:/etcd-data \
  --name etcd_container \
  gcr.io/etcd-development/etcd:${ETCD_VERSION} \
  /usr/local/bin/etcd --data-dir=/etcd-data --name etcd-node-0 \
  --enable-v2=true \
  --initial-advertise-peer-urls http://${VM1_PRIVATE_IP}:2380 \
  --listen-peer-urls http://0.0.0.0:2380 \
  --advertise-client-urls http://${VM1_PRIVATE_IP}:2379 \
  --listen-client-urls http://0.0.0.0:2379 \
  --initial-cluster etcd-node-0=http://${VM1_PRIVATE_IP}:2380 \
  --initial-cluster-state new \
  --initial-cluster-token my-etcd-token

sudo docker exec etcd_container /usr/local/bin/etcdctl endpoint health
```

## Machines

**Machines used in the paper:** All the experiments in the main body of our paper are conducted on 16 AWS p4d.24xlarge instances with 128 A100 GPUs. The number of parameters in the evaluated models is 100 billion.

**Minimal working example:** In this AE, we aim at minimal working examples with 32 V100 GPUs in 4 AWS p3dn.24xlarge instances with auto scaling group (ASG) for the model training.
The number of parameters in models is around 5 billion. A larger model size might cause out-of-memory issues and crash training.

**Use our machines:** We can provide 4 AWS p3dn.24xlarge instances for AE in our GPU cluster. Please contact us if needed.


## Installation

Install the code on each machine before running any experiments. Please make sure the machines can successfully install [DeepSpeed](https://github.com/microsoft/DeepSpeed).
You can configure one machine and then use it as a template to initiate new machines in ASG. 
The code needs to be installed at the exact same path on all machines. 


If AEC members directly use our machines for the evaluation, we will have all dependencies pre-installed on all machines.

On a blank VM, install the basic tools before running `git clone`.

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y build-essential git python3-dev python3-pip python3-venv openssh-client openssh-server docker.io pdsh
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
```

Amazon Linux:

```bash
sudo yum update -y
sudo yum groupinstall -y "Development Tools"
sudo yum install -y git python3 python3-devel python3-pip openssh-clients openssh-server docker pdsh
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
```

Log out and back in after adding the user to the `docker` group. Then clone and install the artifact:

```bash
git clone https://github.com/zhuangwang93/SOSP-30_AE.git
cd SOSP-30_AE
python3 -m pip install --upgrade pip
python3 -m pip install torch==1.13.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
python3 -m pip install -r requirements/requirements.txt
python3 -m pip install transformers==4.24.0 boto3
python3 -m pip install -e .
export PATH="$HOME/.local/bin:$PATH"
```

## Two-VM distributed setup

The following steps are for a minimal two-VM run without AWS Auto Scaling Group automation. Run the commands from the same Linux user on both VMs, and keep the repository at the exact same filesystem path on both machines.

### 1. Set up passwordless SSH between the VMs

DeepSpeed launches remote workers through SSH, so each VM must be able to SSH to the other VM without a password prompt.

On VM 1 and VM 2:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
```

Copy the public key from each VM to the other VM:

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub USER@VM1_PRIVATE_IP
ssh-copy-id -i ~/.ssh/id_ed25519.pub USER@VM2_PRIVATE_IP
```

Verify passwordless SSH in both directions:

```bash
ssh USER@VM1_PRIVATE_IP hostname
ssh USER@VM2_PRIVATE_IP hostname
```

If you use private IPs, make sure the VMs are in the same network/security group and that TCP ports 22, 2379, and 2380 are reachable between them. Port 22 is for SSH, and ports 2379/2380 are for etcd.

### 2. Download system requirements

Install CUDA 11.6, PyTorch 1.13.0 with the CUDA 11.6 wheel, and NCCL 2.14.3 on both VMs. If you did not already run the blank-VM bootstrap commands above, install Git, Python, SSH, Docker, and build tools now.

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y build-essential git python3-dev python3-pip python3-venv openssh-client openssh-server docker.io pdsh
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
```

Amazon Linux:

```bash
sudo yum update -y
sudo yum groupinstall -y "Development Tools"
sudo yum install -y git python3 python3-devel python3-pip openssh-clients openssh-server docker pdsh
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
```

Log out and back in after adding the user to the `docker` group.

### 3. Clone, download Python requirements, and install this artifact

Use the same path on both VMs. For example:

```bash
mkdir -p ~/zhuang
cd ~/zhuang
git clone https://github.com/zhuangwang93/SOSP-30_AE.git Gemini
cd Gemini
python3 -m pip install --upgrade pip
python3 -m pip install torch==1.13.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
python3 -m pip install -r requirements/requirements.txt
python3 -m pip install transformers==4.24.0 boto3
python3 -m pip install -e .
export PATH="$HOME/.local/bin:$PATH"
ds_report
```

`boto3` is only needed by `deepspeed/runtime/snapshot/launch.py`; install it even for manual two-VM setup because the launcher imports it at startup.

If you are using this repository checkout instead of cloning from GitHub, copy it to the same path on both VMs and then run `python3 -m pip install -e .` from that path on each VM.

### 4. Create `examples/hostfile`

Create `examples/hostfile` on both VMs with one line per VM. Set `slots` to the number of GPUs to use on that VM.

```bash
cat > examples/hostfile <<'EOF'
VM1_PRIVATE_IP slots=NUM_GPUS_ON_VM1
VM2_PRIVATE_IP slots=NUM_GPUS_ON_VM2
EOF
```

For two 8-GPU VMs:

```bash
cat > examples/hostfile <<'EOF'
10.0.0.10 slots=8
10.0.0.11 slots=8
EOF
```

The ASG helper can also generate this file automatically when running on AWS instances that are in an Auto Scaling Group:

```bash
cd deepspeed/runtime/snapshot
python3 launch.py -m instances -i 2 -g 8 -e ~/zhuang/Gemini/examples
```

### 5. Start etcd

Choose one VM to host etcd. On that VM, replace `VM1_PRIVATE_IP` with the private IP of the etcd VM. Use `sudo docker` unless you have already added your user to the `docker` group and opened a new login shell.

```bash
# Replace this with the private IP of the VM that will run etcd.
export VM1_PRIVATE_IP=10.0.0.10
export ETCD_VERSION=v3.5.0

sudo docker rm -f etcd_container || true
sudo docker volume create --name etcd-data
sudo docker run --rm -d \
  -p 2379:2379 -p 2380:2380 \
  -v etcd-data:/etcd-data \
  --name etcd_container \
  gcr.io/etcd-development/etcd:${ETCD_VERSION} \
  /usr/local/bin/etcd --data-dir=/etcd-data --name etcd-node-0 \
  --enable-v2=true \
  --initial-advertise-peer-urls http://${VM1_PRIVATE_IP}:2380 \
  --listen-peer-urls http://0.0.0.0:2380 \
  --advertise-client-urls http://${VM1_PRIVATE_IP}:2379 \
  --listen-client-urls http://0.0.0.0:2379 \
  --initial-cluster etcd-node-0=http://${VM1_PRIVATE_IP}:2380 \
  --initial-cluster-state new \
  --initial-cluster-token my-etcd-token

sudo docker exec etcd_container /usr/local/bin/etcdctl endpoint health
```

For ASG-based runs, the launcher can start etcd over SSH:

```bash
cd deepspeed/runtime/snapshot
python3 launch.py -m etcd -i 2 -c 1
```

### 6. Run distributed GPT training

From VM 1, run the tiny GPT distributed smoke test:

```bash
cd ~/SOSP-30_AE
bash examples/GPT/launch.sh
```

`examples/GPT/launch.sh` defaults to `tiny_gpt_template.json`, runs 30 steps, and can be run from either the repository root or `examples/GPT`. The tiny config uses a low initial fp16 loss scale so the run reaches the interleave profiling and snapshot strategy path. To run the larger 5B GPT config instead, pass overrides:

```bash
cd ~/SOSP-30_AE
TRAIN_CONFIG=5B_template.json \
JOB_NAME=GPT2 \
MAX_STEPS=30 \
COMM_PROFILE_STEPS=18 \
JUMP_PROFILE_LINES=8 \
SNAPSHOT_BUFFER_SIZE=32 \
MAX_BLOCKS_IN_SPAN=16 \
LOG_FILE=log_5B \
bash examples/GPT/launch.sh
```

The tiny config keeps the same training path and ZeRO stage as the larger GPT configs, but uses the smallest GPT dimensions that still work with the 512-token fake dataset and GPT-2 tokenizer.

## Code Structure

**Main code:** The system is built upon DeepSpeed and its main code is under [snapshot](deepspeed/runtime/snapshot/). We also hack `deepspeed/runtime/zero/stage3.py` and `deepspeed/runtime/zero/partitioned_param_coordinator.py` to enable the checkpoint of optimizer states for every iteration.

**Examples:** The examples are under [examples](examples/). Three models are used for evaluation: GPT, BERT, and Roberta.

**AE scripts:** The scripts to run the artifact evaluation are under [SOSP_AE](examples/SOSP_AE).


## How to run

You can follow the instructions for evaluations if you'd like to run the code on your machines.

```bash
# Note: the path under our testbed is ~/zhuang/Gemini
# cd SOSP-30_AE
# Step 1: replace the IP addresses of the machines in examples/hostfile, which follows the format of hostfile used in MPI. 
# If you are using ASG for the instances, you can also automatically set the IP addresses with
cd deepspeed/runtime/snapshot
python3 launch.py -m instances

# Step 2: start etcd as the distributed key-value store
cd deepspeed/runtime/snapshot
# If you are using ASG for the instances (strongly recommended), you can start etcd with
python3 launch.py -m etcd
# Otherwise, you can arbitrarily choose one machine from the machines involved in training to set up etcd by setting its IP address as IP1
python3 launch.py -m etcd_ip --etcd-ips "IP1"

# Step 3: run the model script.
# Note that we also provide a script to run all experiments with one command in the next section.
# Note: the path under our testbed is ~/zhuang/Gemini/examples/model_name
cd examples/model_name
bash launch.sh
```


## Artifact Evaluation

### The main claim

**Our main claim:** `Our system can checkpoint the model states for every iteration and it incurs negligible overhead on the training throughput`.

The figures that can demonstrate this claim are Figure 7 (iteration time) and Figure 8 (the network idle time). 

```bash
# Note: the path under our testbed is ~/zhuang/Gemini/examples/SOSP_AE
cd examples/SOSP_AE
# It will take about 30 min to finish the experiments of the three models.
bash run_all.sh
```
The raw data, including both the iteration time of each step and the network idle time, is stored in `results.json`.
The first 8 step times are for warm-up; the middle 10 step times are without any checkpoints; the remaining step times are with GEMINI for checkpointing.
In addition, the generated checkpoints are stored under `model_name/snapshot/`. `x.pt` is the local checkpoint for `Rank x` and `x_y.pt` is the remote checkpoint for `Rank y` stored in `Rank x`.  

After running the script, Figure_7 and Figure_8 will appear in the folder. 
Because of the different experimental settings, the absolute values in these figures may vary from those provided in the paper. 
But you will see that the iteration time is almost the same without checkpoints and with GEMINI for checkpointing.
You can also set `--snapshot_mode` in launch.sh to `naive` from `interleave` to see how traffic blocking for checkpointing affects the iteration time.

### Ablation study

The data in Figure 9, Figure 11, and Figure 14 are collected from simulations. We also provide the simulation code in AE_figures.ipynb. 
You can play with them and the figures will be automatically generated.
