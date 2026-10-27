FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime AS deps-resolver

RUN apt update && \
    apt install -y git ffmpeg build-essential

RUN pip install --upgrade pip 

COPY requirements.txt ./

RUN pip install -r requirements.txt

WORKDIR /app

COPY app ./app

# The commit this image is built from, returned by GET /:
#   docker build --build-arg VERSION=$(git rev-parse --short HEAD) .
# A VERSION set in the deployment overrides it.
ARG VERSION=
ENV VERSION=${VERSION}
