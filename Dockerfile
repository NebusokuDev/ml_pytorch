FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04
LABEL authors="Summit"

ENTRYPOINT ["top", "-b"]

RUN pip3 install