FROM ubuntu
  
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update

RUN apt-get install -y --fix-missing \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    python3-numpy \
    python3-pip \
    python3-opencv \
    software-properties-common \
    zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN pip3 install torch torchvision

WORKDIR /root
COPY caption_app.py .
COPY models.py .
COPY utils.py .
COPY BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar caption_model.pth.tar

#CMD [ "python3", "./image_writer.py" ]
