FROM tensorflow/tensorflow:2.1.0-gpu-py3
WORKDIR /code/
ADD ./requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
ADD ./keras-contrib ./keras-contrib
RUN pip3 install -r keras-contrib
ADD . .

