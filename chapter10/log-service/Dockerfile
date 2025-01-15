FROM python:2.7

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple kafka-python

COPY . /log-service
WORKDIR /log-service
RUN pip install -r requirements.txt

ENTRYPOINT ["python"]
CMD ["save_log.py"]
