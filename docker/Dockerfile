FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

RUN apt-get update \
 && apt-get install --assume-yes --no-install-recommends --quiet \
        python3 \
        python3-pip \
        libz-dev \
        wget

RUN /bin/bash -c "source root/.bashrc"

WORKDIR /agatha_ae