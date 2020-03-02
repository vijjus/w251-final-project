FROM ubuntu

# build this: e.g docker build -t fwriter -f Dockerfile
# to run this container, do:
# on the host:
# make sure you're in the X environment
# xhost + 
# docker run --rm --privileged -e DISPLAY -v /tmp:/tmp -ti face bash

RUN apt update

ENV DEBIAN_FRONTEND=noninteractive

RUN apt install -y python3-opencv python3-pip vim-tiny mosquitto-clients libopencv-dev
RUN apt install -y iputils-ping net-tools
RUN pip3 install paho-mqtt
RUN pip3 install --upgrade s3cmd

WORKDIR /root
COPY .s3cfg .s3cfg
COPY image_writer.py .

CMD [ "python3", "./image_writer.py" ]
