FROM ubuntu:latest
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
MAINTAINER FYLSunghwan "sunghwan519@hotmail.com"
RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev ffmpeg openjdk-11-jdk
COPY . /app
WORKDIR /app
RUN pip3 install Cython numpy
RUN pip3 install -f https://download.pytorch.org/whl/torch_stable.html torchserve torch-model-archiver
RUN sh clean.sh
ENTRYPOINT ["torchserve"]
CMD ["--start", "--model-store", "model_store", "--models", "chordrecog.mar"]

EXPOSE 3000