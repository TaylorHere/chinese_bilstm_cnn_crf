FROM tensorflow/tensorflow:2.1.0-gpu-py3
WORKDIR /code/
ADD ./requirements.txt requirements.txt
RUN git clone https://www.github.com/keras-team/keras-contrib.git && pip3 install keras-contrib/
RUN pip3 install -r requirements.txt
ADD . .

