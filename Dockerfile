FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04
ENTRYPOINT ["top", "-b"]

RUN pip3 install pip3