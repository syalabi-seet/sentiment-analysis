# Dockerfile for cloud v1

FROM nvcr.io/nvidia/tensorrt:21.03-py3

RUN apt-get update && apt install -y \
    nvidia-utils-450 \
    ubuntu-drivers-common

RUN apt update && \
    apt-get install -y \
    libpq-dev \
    gcc

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

LABEL maintainer="Xaltius"

ENV PYTHONUNBUFFERED=TRUE

ENV TOKENIZERS_PARALLELISM=false

RUN mkdir -p /opt/ml/processing/model
RUN mkdir -p /opt/ml/processing/output

COPY . /usr/src/app