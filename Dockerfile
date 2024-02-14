FROM pytorch/torchserve:0.9.0-cpu

ENV MODEL_NAME=ENTER_MODEL_NAME_HERE

WORKDIR /home/model-server/

COPY ${MODEL_NAME}.mar .

CMD ["torchserve", "--start", "--model-store", "/home/model-server", "--models", "model=<MODEL_NAME>.mar"]