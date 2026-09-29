FROM nvidia/cuda:12.8.1-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev build-essential ca-certificates \
    libgl1 libglib2.0-0
RUN python3 -m venv /opt/icshm-runtime
ENV PATH=/opt/icshm-runtime/bin:$PATH
COPY requirements.txt /opt/icshm-reproduction-requirements.txt
RUN python -m pip install --no-cache-dir --timeout 30 --retries 1 \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    -r /opt/icshm-reproduction-requirements.txt
ENV PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
WORKDIR /work
COPY runtime_extras.txt /opt/icshm-runtime-extras.txt
RUN python -m pip install --no-cache-dir --timeout 30 --retries 1 \
    -r /opt/icshm-runtime-extras.txt
