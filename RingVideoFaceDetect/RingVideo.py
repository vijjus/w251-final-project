#
# Update to take an input file and output file
#
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch
import cv2, mmcv
# confirm mtcnn was installed correctly
from facenet_pytorch import MTCNN, extract_face # use pytorch based MTCNN
# from mtcnn import MTCNN
import numpy as np
import face_recognition
import os
from time import sleep
import sys

# Test for GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# set detector
# mtcnn = MTCNN(keep_all=True)
mtcnn = MTCNN(
    image_size=160, margin=20, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=False,
    device=device
)

#filename = "./package.mp4"
#filename = "./RingVideo_1.mp4"
filename = sys.argv[1]
print("Reading from " + filename)

#filename_write = "./FamilyID_2.avi"
filename_write = "./output_movie/" + sys.argv[2]
print("Writting to " + filename_write)

# Open the input movie file
video = mmcv.VideoReader(filename)
# video = cv2.VideoCapture(filename)
frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]

# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'MJPG') # Was XVID
output_movie = cv2.VideoWriter(filename_write, fourcc, 15, (1920, 1080), True)

def match_face_encodings(ke, ue):
    results = face_recognition.compare_faces([ke], ue)
    if results[0] == True:
        return True
    else:
        return False

def create_known_embedding(image_path):
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]
    person = image_path.split('/')[-1].split('.')[0]
    return (person, encoding)

KNOWN_FACES = './data/known_faces/'

files = os.listdir(KNOWN_FACES)

known_people = []
known_encodings = []
for file in files:
    person, encoding = create_known_embedding(KNOWN_FACES + file)
    known_people.append(person)
    known_encodings.append(encoding)
# print(known_people)

cropped_faces = []
person_list = []
count = 0
frame_number = 0
for k, frame in enumerate(frames):
#    print(type(frame))
#    im_np = np.asarray(frame)
    bgr_frame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
    # Show a Progress Bar
    n = len(frames)
    j = (k + 1) / n
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
    sys.stdout.flush()
    #sleep(0.25)

    # Detect faces
    #vboxes, probs, points = mtcnn.detect(img, landmarks=True)
    boxes, p = mtcnn.detect(frame)
    frame_number += 1
    for i in range(len(p)):
        if p[i] is not None and p[i] > 0.9:
#            print("Person detected in frame: {}".format(k))
            count += 1
            face_crop = frame.crop(boxes[i])
            arr_img = np.array(face_crop)
            bgr_image = cv2.cvtColor(arr_img, cv2.COLOR_RGB2BGR)
            image_name = ("./output_faces/test" + str(count) + ".jpg")
            res = cv2.imwrite(image_name, bgr_image)
            if not res:
                continue
            img = face_recognition.load_image_file(image_name)
            enc = face_recognition.face_encodings(img)
            # image not clear enough for encoding
            # save the image for future analysis
            if len(enc) == 0:

                left = boxes[i][0]
                bottom = boxes[i][1]
                right = boxes[i][2]
                top = boxes[i][3]

                # Draw a box around the face
                bgr_frame = cv2.rectangle(bgr_frame, (left, top), (right, bottom), (0, 0, 255), 2)

                continue
            # Check all faces at once
            distances = face_recognition.face_distance(known_encodings, enc[0])
            match_index = np.where(distances == np.amin(distances))[0]
            if distances[match_index] < 0.6:
#                print("{0} is at the door!! Image{1}".format(known_people[match_index.item(0)], count))
                name = known_people[match_index.item(0)]
                image_person = known_people[match_index.item(0)] + str(count)
                person_list.append(image_person)
#                person_list.append(known_people[match_index.item(0)])

                left = boxes[i][0]
                bottom = boxes[i][1]
                right = boxes[i][2]
                top = boxes[i][3]

                # Draw a box around the face
                bgr_frame = cv2.rectangle(bgr_frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # Draw a label with a name below the face
                bgr_frame = cv2.rectangle(bgr_frame, (left, int(bottom - 25)), (right, bottom), (0, 255, 0), -1)
                font = cv2.FONT_HERSHEY_DUPLEX
                bgr_frame = cv2.putText(bgr_frame, name, (int(left + 6), int(bottom - 6)), font, 0.5, (255, 255, 255), 1)
            else:
                left = boxes[i][0]
                bottom = boxes[i][1]
                right = boxes[i][2]
                top = boxes[i][3]

                # Draw a box around the face
                bgr_frame = cv2.rectangle(bgr_frame, (left, top), (right, bottom), (255, 0, 0), 2)

                # Draw a label with a name below the face
                bgr_frame = cv2.rectangle(bgr_frame, (left, int(bottom - 25)), (right, bottom), (255, 0, 0), -1)
                font = cv2.FONT_HERSHEY_DUPLEX
                bgr_frame = cv2.putText(bgr_frame, "unknown", (int(left + 6), int(bottom - 6)), font, 0.5, (255, 255, 255), 1)

    # Write the resulting image to the output video file
    output_movie.write(bgr_frame)

#return count, person_list
# List images that were ID'd and the image #
print(person_list)

# All done!
output_movie.release()
