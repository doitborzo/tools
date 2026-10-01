#!/usr/bin/env bash
# Standalone: installs everything, downloads DeepSeek V4 Flash, runs it.
# No other files needed. 4 GPUs, no CPU offloading.
#
# Follows https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/discussions/44
# (that report is 2 GPUs; this is the same recipe with data-parallel 4).
#
# Usage:  bash run_deepseek.sh
# Runs in the foreground. To background it:  nohup bash run_deepseek.sh &

set -euo pipefail

MODEL="RedHatAI/Muse-Glimmer-30B-FP8-block"
SERVED_NAME="muse-glimmer"
PORT=8000
GPUS="0"
MAX_MODEL_LEN=131072
# LoRA adapters as name=path, space-separated; cowbench/run_tests.sh sets it.
LORA_MODULES="${LORA_MODULES:-pose638=/workspace/lora-runs/lora2pose_w896/adapters/step-00638}"
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"
# Share of the GPU's memory vLLM takes. cowbench/run_tests.sh lowers it to
# leave ~4 GB for the cow detector, which runs on the same card.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"

# "auto" resolves the torch build from the driver. It is right only when uv is
# recent enough: uv 0.9.0 did not know CUDA 13 and quietly installed
# torch+cu126, whose kernels stop at sm_90 - two generations below an
# RTX PRO 6000 Blackwell (sm_120). The uv self-update in step 4 is what makes
# "auto" trustworthy again; override with TORCH_BACKEND=cu130 if it is not.
TORCH_BACKEND="${TORCH_BACKEND:-auto}"

# On /workspace, not $HOME: a pod's home directory is on the ephemeral root
# overlay and is wiped on restart. The venv is several GB and the checkpoint
# is ~167 GiB - neither is worth re-fetching every time.
export HF_HOME=/workspace/hf-cache
export VLLM_CACHE_ROOT=/workspace/vllm-cache
VENV=/workspace/serving/.venv

echo "=============================================================="
echo " 1/6  Open-file limit"
echo "=============================================================="
# THIS IS THE ONE THAT BIT US. Data-parallel opens a coordinator, 4 API
# servers, 4 engine cores and several ZMQ sockets per rank. At the default
# 1024 the coordinator dies while creating sockets - before it can report
# anything - and the only error you get is:
#   RuntimeError: DP Coordinator process failed to report ZMQ addresses
#   within timeout=120 seconds during startup
ulimit -n 65536 2>/dev/null || true
echo "open files: soft=$(ulimit -Sn) hard=$(ulimit -Hn)"
if [ "$(ulimit -Sn)" != "unlimited" ] && [ "$(ulimit -Sn)" -lt 8192 ]; then
    echo ""
    echo "!! Only $(ulimit -Sn) file descriptors, and the hard limit blocks raising it."
    echo "!! Data-parallel will almost certainly fail here. Recreate the pod with"
    echo "!!   --ulimit nofile=65536"
    echo "!! (or the equivalent in the RunPod template) and run this again."
    echo ""
    read -r -p "Continue anyway? [y/N] " reply
    [ "$reply" = "y" ] || exit 1
fi

echo "=============================================================="
echo " 2/6  Clearing GPUs"
echo "=============================================================="
# nvidia-smi is authoritative - killing "vllm serve" alone misses the renamed
# children (EngineCore_DP*, ApiServer_*, DPCoordinator, VLLM::*), and a
# leftover one holds the ZMQ addresses the next run needs.
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -v '^$' | xargs -r kill -9 || true
for pat in "vllm serve" "VLLM::" "EngineCore" "ApiServer" "DPCoordinator"; do
    pkill -9 -f "$pat" 2>/dev/null || true
done
sleep 5
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

echo "=============================================================="
echo " 3/6  CUDA toolkit"
echo "=============================================================="
# Needed at RUNTIME, not to build anything: vLLM JIT-compiles FlashInfer,
# DeepGEMM and Triton kernels on first use and calls nvcc to do it.
CUDA_WANT="13.4"

# Duplicate NVIDIA apt sources make EVERY apt command fail, purge included:
#   E: Conflicting values set for option Signed-By regarding source
#      .../cuda/repos/ubuntu2404/x86_64/ : /usr/share/keyrings/cuda-archive-keyring.gpg !=
#   E: The list of sources could not be read.
# These images often ship the CUDA repo added the old apt-key way (no
# Signed-By), and installing cuda-keyring adds a second entry for the same
# repo WITH Signed-By. apt refuses to pick between them. Keep the keyring
# package's own file and disable any other entry for that repo.
cuda_fix_apt_sources() {
    local distro="$1" canonical="/etc/apt/sources.list.d/cuda-${distro}-x86_64.list"
    local f changed=0
    while IFS= read -r f; do
        [ "$f" = "$canonical" ] && continue
        if [ "$f" = "/etc/apt/sources.list" ]; then
            # Never disable the whole file - comment out just the cuda lines.
            sed -i '\|developer\.download\.nvidia\.com/compute/cuda|s|^|#|' "$f"
            echo "  commented out CUDA lines in $f"
        else
            mv "$f" "$f.disabled"
            echo "  disabled duplicate source: $f -> $f.disabled"
        fi
        changed=1
    done < <(grep -rl "developer\.download\.nvidia\.com/compute/cuda" \
                 /etc/apt/sources.list /etc/apt/sources.list.d/ 2>/dev/null \
             | grep -v '\.disabled$' || true)
    [ "$changed" -eq 0 ] && echo "  no duplicate NVIDIA apt sources"
    return 0
}

distro="$(. /etc/os-release && echo "${ID}${VERSION_ID}" | tr -d '.')"
echo "checking apt sources for duplicate NVIDIA repos"
cuda_fix_apt_sources "$distro"


cuda_installed_version() {
    local nvcc
    nvcc="$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)"
    [ -x "$nvcc" ] || return 1
    # "Cuda compilation tools, release 13.0, V13.0.88" -> 13.0
    "$nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1
}

current="$(cuda_installed_version || true)"

# sort -V puts the lower version first; if that is the wanted one, the
# installed toolkit is at least as new and there is nothing to do.
if [ -n "$current" ] && [ "$(printf '%s\n%s\n' "$CUDA_WANT" "$current" | sort -V | head -1)" = "$CUDA_WANT" ]; then
    echo "CUDA toolkit $current already installed (>= $CUDA_WANT), keeping it"
else
    if [ -n "$current" ]; then
        echo "CUDA toolkit $current is older than $CUDA_WANT - removing it first"
        # Toolkit packages ONLY. Driver packages are excluded deliberately:
        # on these pods the driver comes from the host, and removing it would
        # take the GPUs away entirely - something no reinstall here could undo.
        mapfile -t to_purge < <(
            dpkg-query -W -f='${Package}\n' 2>/dev/null \
            | grep -E '^(cuda-toolkit|cuda-compiler|cuda-command-line-tools|cuda-nvcc|cuda-cudart|cuda-libraries|cuda-nvtx|cuda-nvml-dev|cuda-nvprof|cuda-cccl|cuda-crt|cuda-nvdisasm|cuda-nvvm|cuda-profiler|cuda-sanitizer|cuda-documentation|cuda-nsight|cuda-gdb|libcublas|libcufft|libcurand|libcusolver|libcusparse|libnpp|libnvjitlink|libnvjpeg|libcufile)' \
            | grep -vE 'nvidia-driver|libnvidia-(compute|gl|decode|encode|extra|cfg|common)|cuda-drivers' || true
        )
        if [ "${#to_purge[@]}" -gt 0 ]; then
            printf '  purging: %s\n' "${to_purge[@]}"
            apt-get purge -y -qq "${to_purge[@]}" || true
            apt-get autoremove -y -qq || true
        else
            echo "  no CUDA toolkit apt packages found - it was probably installed"
            echo "  from a runfile, so /usr/local/cuda-$current is left in place"
        fi
        # The /usr/local/cuda symlink survives a purge and would keep pointing
        # at the version just removed, so nvcc lookups resolve to nothing.
        [ -L /usr/local/cuda ] && rm -f /usr/local/cuda
    fi

    echo "installing cuda-toolkit-${CUDA_WANT//./-} for $distro"
    curl -fsSL -o /tmp/cuda-keyring.deb \
        "https://developer.download.nvidia.com/compute/cuda/repos/${distro}/x86_64/cuda-keyring_1.1-1_all.deb"
    dpkg -i /tmp/cuda-keyring.deb
    apt-get update -qq
    apt-get install -y -qq "cuda-toolkit-${CUDA_WANT//./-}"

    # A fresh install can leave the generic symlink missing (we may have just
    # deleted it above), which breaks CUDA_HOME below.
    if [ ! -e /usr/local/cuda ] && [ -d "/usr/local/cuda-${CUDA_WANT}" ]; then
        ln -sfn "/usr/local/cuda-${CUDA_WANT}" /usr/local/cuda
    fi
fi

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo "=============================================================="
echo " 4/6  uv + venv + vLLM"
echo "=============================================================="
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# The installer writes to ~/.local/bin, which a non-login shell may not have
# on PATH; the guide re-execs the shell, sourcing this env file is equivalent.
[ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"
export PATH="$HOME/.local/bin:$PATH"
# uv itself is installed once and then never updated, which is how a uv old
# enough to be unaware of CUDA 13 survived on this pod and resolved
# --torch-backend=auto to cu126. "uv self update" is a silent no-op when uv
# came from a package manager or from an installer channel that is itself
# pinned - on this pod it left 0.9.0 in place - so check whether the version
# actually moved and re-run the standalone installer when it did not.
uv_before="$(uv --version 2>/dev/null || echo none)"
uv self update 2>/dev/null || true
if [ "$(uv --version 2>/dev/null || echo none)" = "$uv_before" ]; then
    echo "  uv self update did nothing ($uv_before) - reinstalling via installer"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"
    hash -r
fi
uv --version

if [ ! -d "$VENV" ]; then
    mkdir -p "$(dirname "$VENV")"
    uv venv --python 3.12 --seed --managed-python "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# "can I import vllm" is not enough: the venv that fails here imports fine and
# only dies on the first kernel launch, deep inside engine startup, with
#   CUDA error: no kernel image is available for execution on the device
# The real question is whether this GPU's compute capability is in the arch
# list torch was BUILT for, so ask that instead.
venv_ready() {
    python - <<'PY' 2>/dev/null
import sys

try:
    import torch
    import vllm  # noqa: F401
except Exception:
    sys.exit(1)

try:
    major, minor = torch.cuda.get_device_capability()
except Exception:
    sys.exit(1)

sys.exit(0 if f"sm_{major}{minor}" in torch.cuda.get_arch_list() else 1)
PY
}

# Last resort, and the only step here that does not depend on uv knowing this
# CUDA version at all: pull the torch trio straight from the PyTorch index for
# CUDA 13.0, pinned to the exact versions vLLM already resolved. This is
# verbatim what torch's own "does not include kernels for this GPU" warning
# tells you to do.
#
# --reinstall-package, not --reinstall: the latter would re-resolve all ~200
# packages against the PyTorch index, which does not carry most of them.
force_torch_cuda_index() {
    local idx="$1" pkg ver pins=()
    for pkg in torch torchvision torchaudio; do
        ver="$(python -c "import importlib.metadata as m, sys; sys.stdout.write(m.version('$pkg').split('+')[0])" 2>/dev/null || true)"
        [ -n "$ver" ] && pins+=("$pkg==$ver" --reinstall-package "$pkg")
    done
    [ "${#pins[@]}" -gt 0 ] || return 1
    echo "  reinstalling torch from $idx"
    uv pip install --index-url "$idx" --extra-index-url https://pypi.org/simple "${pins[@]}"
}

if ! venv_ready; then
    # --reinstall, not a bare install: an existing venv holding the wrong
    # torch is exactly the case this branch exists for, and uv would otherwise
    # consider the requirement already satisfied and change nothing.
    echo "installing vllm (torch backend: $TORCH_BACKEND)"
    uv pip install --reinstall vllm --torch-backend="$TORCH_BACKEND"

    # "auto" is a guess made by uv from the driver, and an old uv guesses cu126
    # on hardware that cannot run it. Name the backend explicitly before
    # falling back to the index URL. Tolerate failure: a uv that predates
    # CUDA 13 rejects cu130 as an invalid value for the flag.
    if ! venv_ready && [ "$TORCH_BACKEND" = "auto" ]; then
        echo "  auto resolved to an unusable torch - retrying with cu130"
        uv pip install --reinstall vllm --torch-backend=cu130 || true
    fi

    if ! venv_ready; then
        force_torch_cuda_index https://download.pytorch.org/whl/cu130 || true
    fi

    # Verify rather than assume, before the 33 GB checkpoint download and the
    # engine startup that would otherwise find this out the slow way.
    if ! venv_ready; then
        echo ""
        echo "!! The venv is still not usable after installing."
        python - <<'PY' || true
import torch
import vllm
print("   vllm     :", vllm.__version__)
print("   torch    :", torch.__version__, "cuda", torch.version.cuda)
print("   arch list:", torch.cuda.get_arch_list())
print("   this GPU :", "sm_%d%d" % torch.cuda.get_device_capability())
PY
        echo "!! Neither --torch-backend=cu130 nor the PyTorch cu130 index produced"
        echo "!! a torch with kernels for this GPU. Check that a build of this exact"
        echo "!! torch version exists on the index above, then install it by hand."
        echo "!! (sm_120 / Blackwell needs a CUDA 12.9+ / 13.x build.)"
        exit 1
    fi
fi
python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| arch', torch.cuda.get_arch_list())"

echo "=============================================================="
echo " 5/6  Model"
echo "=============================================================="
command -v hf >/dev/null 2>&1 || uv pip install huggingface_hub
df -h "$HF_HOME" | tail -1
# Resumes and no-ops if already complete.
hf download "$MODEL"

echo "=============================================================="
echo " 6/6  Starting vLLM"
echo "=============================================================="
# export OMP_NUM_THREADS=8
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export CUDA_VISIBLE_DEVICES="$GPUS"

# Notes on the flag set, all learned the hard way:
#   --data-parallel-size + --enable-expert-parallel
#       Not tensor-parallel. For a MoE this shards experts across ranks
#       instead of splitting every matmul, which is what makes the report's
#       278 tok/s possible on PCIe hardware with no NVLink.
#   --kv-cache-dtype fp8
#       Mandatory, not a memory choice: DeepSeek V4's sparse-MLA attention
#       only implements an fp8 cache layout and asserts during layer
#       construction on anything else.
#   --disable-custom-all-reduce
#       vLLM's custom all-reduce assumes fast peer-to-peer links.
#   no --host
#       Deliberately absent. The default binds fine for an SSH tunnel
#       (which connects to 127.0.0.1:PORT inside the pod).
#   no --kv-offloading-*
#       No CPU offloading, as asked. It would spill KV cache to host RAM
#       over PCIe on every token.
#
# If startup fails, the parsers on the next line are the first thing to
# drop - they are the only flags here the linked report does not use.
exec vllm serve "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --generation-config auto \
    --tensor-parallel-size 1 \
    --enable-auto-tool-choice \
    --tool-call-parser muse_glimmer \
    --reasoning-parser muse_glimmer \
    --trust-remote-code \
    --enable-lora --max-lora-rank "$MAX_LORA_RANK" --max-loras 2 \
    --lora-modules $LORA_MODULES \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --port "$PORT"
