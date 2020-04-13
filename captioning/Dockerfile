FROM nvcr.io/nvidia/pytorch:19.09-py3

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /root
COPY models.py .
COPY utils.py .
COPY BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar caption_model.pth.tar
COPY caption_app.py .
