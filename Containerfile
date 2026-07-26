# Use an NVIDIA CUDA base image
FROM docker.io/nvidia/cuda:13.0.0-devel-ubuntu24.04

# Install Python 3.14 and venv
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y python3.14 python3.14-venv python3.14-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Create venv and prepend its bin directory to PATH to activate it
ENV VIRTUAL_ENV=/opt/venv
RUN python3.14 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install dependencies and make editable
RUN pip install --no-cache-dir .[cuda]
RUN pip install --no-cache-dir -e .

# Allow untrusted code execution (for human-eval)
ENV HF_ALLOW_CODE_EVAL='1'

# Acts as the %runscript in Apptainer
ENTRYPOINT ["secret_loyalties"]
