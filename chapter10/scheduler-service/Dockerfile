FROM ubuntu:16.04

ENV KAFKA_HOST 127.0.0.1
ENV KAFKA_PORT 9092

ENV LOG_HOST 127.0.0.1
ENV LOG_PORT 7777

RUN apt-get update -y
RUN apt-get install -y python-pip python-dev

COPY . /scheduler-service
WORKDIR /scheduler-service

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

ENTRYPOINT ["python3"]
CMD ["run.py"]
