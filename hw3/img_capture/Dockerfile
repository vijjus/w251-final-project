FROM ubuntu

# build this: e.g docker build -t face -f Dockerfile
# to run this container, do:
# on the host:
# make sure you're in the X environment
# xhost + 
# docker run --rm --privileged -e DISPLAY -v /tmp:/tmp -ti face bash

RUN apt update

ENV DEBIAN_FRONTEND=noninteractive

RUN apt install -y python3-opencv python3-pip vim-tiny mosquitto-clients libopencv-dev
RUN pip3 install paho-mqtt

WORKDIR /root
COPY face_capture.py .

CMD ["python3", "./face_capture.py"]
