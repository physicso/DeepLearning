FROM ubuntu:16.04

ENV KAFKA_HOST 192.168.31.100
ENV KAFKA_PORT 9092
ENV TF_SERVING_HOST 192.168.31.100
ENV TF_SERVING_PORT 9000

COPY . /worker-service
WORKDIR /worker-service

RUN apt-get install -y python-pip python-dev
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

WORKDIR /worker-service
CMD sh ./initial.sh