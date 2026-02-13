FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/requirements.txt

RUN git clone https://github.com/VAST-AI-Research/TripoSR.git /opt/TripoSR
ENV PYTHONPATH=/opt/TripoSR:${PYTHONPATH}

COPY src /app/src
COPY convert.py /app/convert.py

RUN mkdir -p /app/input /app/output

ENTRYPOINT ["python", "/app/convert.py"]
CMD ["--help"]
