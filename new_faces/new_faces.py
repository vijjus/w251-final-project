from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
import face_recognition
import os
from time import sleep
import sys
from facenet_pytorch import MTCNN

# Test for GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
    image_size=160, margin=20, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=False,
    device=device
)

def analyze_image(image, encoding, embedding_dict):
    for key, value in embedding_dict.items():
        enc = value[0]
        res = face_recognition.compare_faces([enc], encoding)
        if res:
            print("A matching image found: {}, {}".format(key, image))
            value[1].append(image)
            return
    print("Adding a new image: {}".format(image))
    embedding_dict[image] = (encoding,[image])

def process_directory(img_dir):
    embedding_dict = {}
    # go over the directory
    for f in os.listdir(img_dir):
        # read the image from the file
        np_frame = cv2.imread(img_dir + f)
        if np_frame is None:
            print("Could not read image: {}".format(f))
            continue
        frame = Image.fromarray(np_frame)
        boxes, p = mtcnn.detect(frame)
        for i in range(len(p)):
            if p[i] is not None and p[i] > 0.9:
                print("face detected!!")
                face_crop = frame.crop(boxes[i])
                arr_img = np.array(face_crop)
                bgr_image = cv2.cvtColor(arr_img, cv2.COLOR_RGB2BGR)
                image_name = ("./testimg.jpg")
                res = cv2.imwrite(image_name, bgr_image)
                if not res:
                    print("Could not write cropped face")
                    continue
                print("processing image: {}".format(image_name))
                file_path = image_name
                image = face_recognition.load_image_file(file_path)
                encoding = face_recognition.face_encodings(image)
                if len(encoding) == 0:
                    print("No encoding produced")
                    continue
                #print("{}: {}".format(f, encoding))
                analyze_image(f, encoding[0], embedding_dict)

    for key, value in embedding_dict:
        if len(value[1]) > 1:
            print("images {} are of the same person".format(value[1]))


image_dir = input("Enter the directory containing the images: ")
process_directory(image_dir)
