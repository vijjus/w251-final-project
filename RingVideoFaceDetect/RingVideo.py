#
# Update to take an input file and output file
#
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch
import cv2, mmcv
from facenet_pytorch import MTCNN, extract_face # use pytorch based MTCNN
import numpy as np
import face_recognition
import os
from time import sleep
import sys
import uuid

# Test for GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# set detector
mtcnn = MTCNN(
    image_size=160, margin=20, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=False,
    device=device
)

# Set filename for input video from input argument
try:
    # Set filename for input video from input argument
    filename = "/mnt/video_in/" + sys.argv[1]
    # Set filename for output video from input argument
    filename_write = "/mnt/video_out/" + sys.argv[2]
    # Open the input movie file
    video = mmcv.VideoReader(filename)
except IndexError:
    print('Missing input or output video file name')
    sys.exit(1)
except FileNotFoundError:
    print('No file with that name')
    sys.exit(1)

print("Reading from " + filename)

print("Writting to " + filename_write)

# Read in video as frames for detection and inference
frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]

# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Output as .mp4 format
output_movie = cv2.VideoWriter(filename_write, fourcc, 15, (1920, 1080), True)

# Function for area of box for use on detected face crop size
def area(left, right, bottom, top):
    width = right - left
    height = top - bottom
    area = width * height
    return (area)

# image Encoding Function
def create_known_embedding(image_path):
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]
    person = image_path.split('/')[-1].split('.')[0]
    return (person, encoding)

# Set directory for known face images
KNOWN_FACES = './data/known_faces/'
files = os.listdir(KNOWN_FACES)

# Encode know face images for comparision with detected faces
known_people = []
known_encodings = []
for file in files:
    person, encoding = create_known_embedding(KNOWN_FACES + file)
    known_people.append(person)
    known_encodings.append(encoding)

person_list = []
count = 0
frame_number = 0
frames_list = {}
N = 5 # Set number frames to save for additional processing
key_list = []
for k, frame in enumerate(frames):
    # OpenCV uses BGR colors
    bgr_frame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)

    # Show a progress bar of video processed
    n = len(frames)
    j = (k + 1) / n
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
    sys.stdout.flush()

    # Detect faces
    boxes, p = mtcnn.detect(frame)
    frame_number += 1
    detect_area = 0
    for i in range(len(p)):
        if p[i] is not None and p[i] > 0.9: # Set probabilty threshold
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
            # Set the face crop locations
            left = boxes[i][0]
            bottom = boxes[i][1]
            right = boxes[i][2]
            top = boxes[i][3]
            # Calculate the area of the faces
            face_area = area(left, right, bottom, top)
            detect_area = detect_area + face_area

            if len(enc) == 0: # If Image is not clear enough encoding is empty
                # Draw a red box around the face that connot be encoded
                bgr_frame = cv2.rectangle(bgr_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                continue

            # Check all faces at once
            distances = face_recognition.face_distance(known_encodings, enc[0])
            match_index = np.where(distances == np.amin(distances))[0]

            if distances[match_index] < 0.6:
                name = known_people[match_index.item(0)]
                # image_person = known_people[match_index.item(0)] + str(count)
                # person_list.append(image_person)
                if name not in person_list:
                    person_list.append(name)

                # Draw a box around the face
                bgr_frame = cv2.rectangle(bgr_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                # Draw a label with a name below the face
                bgr_frame = cv2.rectangle(bgr_frame, (left, int(bottom - 25)), (right, bottom), (0, 255, 0), -1)
                font = cv2.FONT_HERSHEY_DUPLEX
                bgr_frame = cv2.putText(bgr_frame, name, (int(left + 6), int(bottom - 6)), font, 0.5, (255, 255, 255), 1)

            else:
                # Draw a box around the face
                bgr_frame = cv2.rectangle(bgr_frame, (left, top), (right, bottom), (255, 0, 0), 2)
                # Draw a label with a name below the face
                bgr_frame = cv2.rectangle(bgr_frame, (left, int(bottom - 25)), (right, bottom), (255, 0, 0), -1)
                font = cv2.FONT_HERSHEY_DUPLEX
                bgr_frame = cv2.putText(bgr_frame, "unknown", (int(left + 6), int(bottom - 6)), font, 0.5, (255, 255, 255), 1)

    # Save N number of frames by total face_area
    if len(frames_list) < N:
        frames_list.update({detect_area : frame})
    else:
        key_list = list(dict.keys(frames_list))
        if detect_area > min(key_list):
            del frames_list[min(key_list)]
            frames_list.update({detect_area : frame})

    # Write the resulting image to the output video file
    output_movie.write(bgr_frame)

if len(person_list) == 0:
    print(frames_list)
    for frame in frames_list.values():
        save_frame = np.array(frame)
        bgr_save_frame = cv2.cvtColor(save_frame, cv2.COLOR_RGB2BGR)
        frame_name = ("./unknown_frames/frame_" + str(uuid.uuid4()) + ".jpg")
        res = cv2.imwrite(frame_name, bgr_save_frame)

else:
    # Print list of identified know people
    print(person_list)

# All done release new video file and remove input file
output_movie.release()
os.remove(filename)
