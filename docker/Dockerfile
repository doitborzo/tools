# syntax=docker/dockerfile:1
# herd in one image: the code, its Python environment and the models (docker/models/*.tgz, unpacked
# into /models at build time). Built from the repository root - docker/compose.yaml does that:
#
#   cd docker && docker compose build                     (TORCH_BACKEND=cpu docker compose build: no NVIDIA card)
#
# Everything the barn and the dataset tools write goes to /data (docker/data on the host).
ARG BASE=python:3.12-slim
FROM ${BASE} AS deps

# cu126: NVIDIA cards up to the RTX 40 series; cu128: RTX 50; cpu: no NVIDIA card (Mac, laptops)
ARG TORCH_BACKEND=cu126
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/hf-cache TZ=Europe/Kyiv

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 libgomp1 tzdata \
    && rm -rf /var/lib/apt/lists/*
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/${TORCH_BACKEND}
# torch stays the build above (rfdetr would otherwise pull another one from PyPI)
RUN pip freeze | grep -iE '^(torch|torchvision)==' > /tmp/torch.txt \
    && pip install -c /tmp/torch.txt "transformers>=5.15" opencv-python-headless numpy scipy pillow requests \
       safetensors rfdetr

FROM deps
# The models: every archive in docker/models/ (herd_models_*.tgz, herd_reid_*.tgz, herd_rumination_*.tgz)
# and every folder there, into /models - read from the build context, so the archives never sit in a layer
RUN --mount=type=bind,source=docker/models,target=/incoming \
    mkdir -p /models && \
    for a in /incoming/*.tgz /incoming/*.tar.gz; do \
        if [ -f "$a" ]; then echo "unpacking $(basename "$a")"; tar -xzf "$a" -C /models; fi; \
    done && \
    for d in /incoming/*/; do if [ -d "$d" ]; then cp -a "$d" /models/; fi; done

COPY herd /opt/herd/herd
COPY cowbench /opt/herd/cowbench
COPY docker/entrypoint.sh docker/warmup.py /opt/herd/docker/

# every model loaded once, with the internet still there: whatever they fetch besides their own
# weights (configs, the detector's backbone) is kept in the image; after this the container is offline
WORKDIR /models
RUN python /opt/herd/docker/warmup.py
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

WORKDIR /data
ENTRYPOINT ["bash", "/opt/herd/docker/entrypoint.sh"]
CMD ["help"]
