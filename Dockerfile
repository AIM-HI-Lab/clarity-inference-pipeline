# CPU-friendly image (runs anywhere). For NVIDIA GPU inference, build Dockerfile.gpu instead.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    nnUNet_raw_data_base=/opt/nnunet/v1/raw \
    nnUNet_preprocessed=/opt/nnunet/v1/preprocessed \
    RESULTS_FOLDER=/opt/nnunet/v1/results \
    TOTALSEG_HOME_DIR=/opt/totalsegmentator

ARG INSTALL_TOTALSEGMENTATOR=1
ARG INSTALL_NNUNET=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    dcm2niix \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/nnunet/v1/raw /opt/nnunet/v1/preprocessed /opt/nnunet/v1/results /opt/totalsegmentator

WORKDIR /app

COPY pyproject.toml README.md LICENSE /app/
COPY src /app/src
COPY docker/entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# setuptools<82 avoids occasional editable-install issues (matches dev/setup_local_models.sh)
RUN python -m pip install --upgrade pip "setuptools<82" wheel && \
    python -m pip install .

RUN if [ "$INSTALL_TOTALSEGMENTATOR" = "1" ]; then python -m pip install TotalSegmentator; fi && \
    if [ "$INSTALL_NNUNET" = "1" ]; then python -m pip install nnunetv2 nnunet; fi

RUN if [ "$INSTALL_TOTALSEGMENTATOR" = "1" ]; then totalseg_download_weights -t total || true; fi

RUN if [ "$INSTALL_NNUNET" = "1" ]; then \
    curl -fsSL -o /tmp/Task135_KiTS2021.zip "https://zenodo.org/records/5126443/files/Task135_KiTS2021.zip?download=1" && \
    nnUNet_install_pretrained_model_from_zip /tmp/Task135_KiTS2021.zip && \
    rm -f /tmp/Task135_KiTS2021.zip; \
    fi

LABEL org.opencontainers.image.title="axis-inference-pipeline" \
      org.opencontainers.image.description="DICOM → TotalSegmentator → nnU-Net tumor → axis-pn (SWP)"

ENTRYPOINT ["/docker-entrypoint.sh"]
