FROM pytorch/torchserve:0.9.0-cpu

WORKDIR /home/model-server/

COPY ${MODEL_NAME}.mar .

CMD ["torchserve", "--start", "--model-store", "/home/model-server", "--models", "model=model.mar"]