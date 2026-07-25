FROM nvidia/cuda:13.3.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies and Python 3.14
RUN apt-get update && \
    apt-get install -y software-properties-common make curl && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y python3.14 python3.14-dev python3.14-venv

# Bootstrap pip for Python 3.14 specifically
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.14

WORKDIR /app
COPY . /app

# Run your build steps using the specific python binary
RUN make install && \
    python3.14 -m pip install .
