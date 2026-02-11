FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime AS deps-resolver

RUN apt update && \
    apt install -y git ffmpeg build-essential

RUN pip install --upgrade pip 

COPY requirements.txt ./

RUN pip install -r requirements.txt

WORKDIR /app

COPY app ./app