#
# Running in virtual env Ring
# Python 3.7.4
#
# pip install torch
# Successfully installed torch-1.4.0
#
# pip install opencv-python
# Successfully installed numpy-1.18.1 opencv-python-4.2.0.32
#
# pip install mmcv
# Installing collected packages: addict, pyyaml, six, mmcv
# Successfully installed mmcv-0.3.2
#
# pip install mtcnn
# Installing collected packages: scipy, keras-preprocessing, h5py, keras-applications, keras, mtcnn
# Successfully installed h5py-2.10.0 keras-2.3.1 keras-applications-1.0.8 keras-preprocessing-1.1.0 mtcnn-0.1.0 scipy-1.4.1
#
# pip install tensorflow
# Successfully installed absl-py-0.9.0 astor-0.8.1 cachetools-4.0.0 certifi-2019.11.28 chardet-3.0.4 gast-0.2.2
# google-auth-1.11.3 google-auth-oauthlib-0.4.1 google-pasta-0.2.0 grpcio-1.27.2 idna-2.9 markdown-3.2.1
# oauthlib-3.1.0 opt-einsum-3.2.0 protobuf-3.11.3 pyasn1-0.4.8 pyasn1-modules-0.2.8 requests-2.23.0
# requests-oauthlib-1.3.0 rsa-4.0 tensorboard-2.1.1 tensorflow-2.1.0 tensorflow-estimator-2.1.0 termcolor-1.1.0
# urllib3-1.25.8 werkzeug-1.0.0 wheel-0.34.2 wrapt-1.12.1
#
# pip install --upgrade setuptools
# Successfully installed setuptools-46.0.0
#
# pip install pillow
# successfully installed pillow-7.0.0
#
# install cmake 3.16.5 from download
#
# pip install face_recognition
# first pip install cmake
# Successfully installed cmake-3.16.3
# pip install dlib
# Successfully installed dlib-19.19.0
# Try again, Successfully installed face-recognition-1.3.0
#
# pip install facenet-pytorch
# Successfully installed facenet-pytorch-2.2.9
#
# pip install torchvision
# Successfully installed torchvision-0.5.0
#
# pip install matplotlib
# Successfully installed cycler-0.10.0 kiwisolver-1.1.0 matplotlib-3.2.0 pyparsing-2.4.6 python-dateutil-2.8.1
#
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch
import cv2, mmcv
# confirm mtcnn was installed correctly
from facenet_pytorch import MTCNN, extract_face # use pytorch based MTCNN
# from mtcnn import MTCNN
# from IPython import display
import numpy as np
import face_recognition
import os

# Test for GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# set detector
# mtcnn = MTCNN(keep_all=True)
mtcnn = MTCNN(
    margin=40, min_face_size=40,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=False,
    device=device
)

#filename = "./package.mp4"
filename = "./RingVideo_1.mp4"
print("got " + filename)

video = mmcv.VideoReader(filename)
frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]

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

# known_people = {}
# known_encodings = []
# for file in files:
#     person, encoding = create_known_embedding(KNOWN_FACES + file)
#     known_people[person] = encoding
#     known_encodings.append(encoding)

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
for k, frame in enumerate(frames):
    # Detect faces
    #vboxes, probs, points = mtcnn.detect(img, landmarks=True)
#    print("frame {}".format(k))
    boxes, p = mtcnn.detect(frame)
    for i in range(len(p)):
        if p[i] is not None and p[i] > 0.9:
#            print("Person detected in frame: {}".format(k))
            person_list.append(k)
            count += 1
            face_crop = frame.crop(boxes[i])
            arr_img = np.array(face_crop)
            image_name = ("./output_faces/test" + str(count) + ".jpg")
            res = cv2.imwrite(image_name, arr_img)
            if not res:
                continue
            img = face_recognition.load_image_file(image_name)
            enc = face_recognition.face_encodings(img)
            # image not clear enough for encoding
            # save the image for future analysis
            if len(enc) == 0:
                continue
            # print(enc[0])
            # Check all faces at once
            distances = face_recognition.face_distance(known_encodings, enc[0])
            match_index = np.where(distances == np.amin(distances))[0]
#            print("distance {}".format(distances))
#            print("min distance {}".format(distances[match_index]))
            if distances[match_index] < 0.6:
                print("{0} is at the door!! Image{1}".format(known_people[match_index.item(0)], count))

#            for m in known_people:
#                print("Checking for {}".format(m))
#                face_enc = known_people[m]
#                results = face_recognition.compare_faces([face_enc], enc[0])
#                distances = face_recognition.face_distance(known_encodings, enc[0])
#                match_index = np.where(distances == np.amin(distances))[0]
#                print(match_index[0])
#                if distances[match_index] < 0.6:
#                    print("{0} is at the door!! Image{1}".format(known_people[match_index], count))
#                if results[0] == True:
#                    print("{} is at the door!!".format(m))
#                    print("{} distances".format(distances))
#                    print("min distance {1} index {2} Image{3}".format(distances[match_index[0]], match_index, count))
#                else:
#                    print("An unknown person at the door!!")
#            cropped_faces.append(face_crop)

#fig=plt.figure(figsize=(12, 12))
#columns = 4
#rows = int(np.ceil(count/columns))
#for i in range(0, count):
#    img = frames[person_list[i]]
#    fig.add_subplot(rows, columns, i+1)
#    plt.imshow(img)
#plt.show()

#                    >>> # Draw boxes and save faces
#                    >>> img_draw = img.copy()
#                    >>> draw = ImageDraw.Draw(img_draw)
#                    >>> for i, (box, point) in enumerate(zip(boxes, points)):
#                    ...     draw.rectangle(box.tolist(), width=5)
#                    ...     for p in point:
#                    ...         draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=10)
#                    ...     extract_face(img, box, save_path='detected_face_{}.png'.format(i))
#                    >>> img_draw.save('annotated_faces.png')

#return count, person_list
