# Turn on video from webcam connected to jetson
# Detect Faces with NN

from PIL import Image
import sys
import os
import urllib
import cv2
import tensorflow.contrib.tensorrt as trt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import tensorflow as tf
import time
import uuid
import wget
from tf_trt_models.detection import download_detection_model, build_detection_graph

# https://github.com/yeephycho/tensorflow-face-detection
url_graph = 'https://github.com/yeephycho/tensorflow-face-detection/blob/master/model/frozen_inference_graph_face.pb?raw=true'
FROZEN_GRAPH_NAME = wget.download(url_graph)

#load the frozen graph
output_dir=''
frozen_graph = tf.GraphDef()
with open(os.path.join(output_dir, FROZEN_GRAPH_NAME), 'rb') as f:
  frozen_graph.ParseFromString(f.read())

# https://github.com/NVIDIA-AI-IOT/tf_trt_models/blob/master/tf_trt_models/detection.py
INPUT_NAME='image_tensor'
BOXES_NAME='detection_boxes'
CLASSES_NAME='detection_classes'
SCORES_NAME='detection_scores'
MASKS_NAME='detection_masks'
NUM_DETECTIONS_NAME='num_detections'

input_names = [INPUT_NAME]
output_names = [BOXES_NAME, CLASSES_NAME, SCORES_NAME, NUM_DETECTIONS_NAME]

# Optimize the frozen graph using TensorRT
trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph,
    outputs=output_names,
    max_batch_size=1,
    max_workspace_size_bytes=1 << 25,
    precision_mode='FP16',
    minimum_segment_size=50
)

# Create Session and Load Grapah
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

tf_sess = tf.Session(config=tf_config)

# use this if you want to try on the optimized TensorRT graph
# Note that this will take a while
# tf.import_graph_def(trt_graph, name='')

# use this if you want to try directly on the frozen TF graph
# this is much faster
tf.import_graph_def(frozen_graph, name='')

tf_input = tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')
tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
tf_num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(1)

# Check if camera opened successfully
if (cap.isOpened()== False): 
    print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    #image_resized = np.array(frame.resize((300, 300)))
    image_resized = cv2.resize(frame, (300, 300))
    #image = np.array(frame)

    scores, boxes, classes, num_detections = tf_sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections], feed_dict={tf_input: image_resized[None, ...]})

    boxes = boxes[0] # index by 0 to remove batch dimension
    scores = scores[0]
    classes = classes[0]
    num_detections = num_detections[0]
    print('detection run')

    # suppress boxes that are below the threshold.. 
    DETECTION_THRESHOLD = 0.50

    # plot boxes exceeding score threshold
    for i in range(int(num_detections)):
        if scores[i] < DETECTION_THRESHOLD:
            continue
        
        print(scores[i])
        print(classes[i])
        # print(boxes[i])

        box = boxes[i] * np.array([image_resized.shape[0], image_resized.shape[1], image_resized.shape[0], image_resized.shape[1]])
        box = box.astype(int)
        print(box)
        cv2.rectangle(image_resized,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
        face = image_resized[box[0]:box[2], box[1]:box[3]]
        # cv2.imwrite('/images_face/face.jpg', face)

        # Create a unique name for the image file
        name = "/images_face/face_" + str(uuid.uuid4()) + ".png"

        # Write image to drive
        cv2.imwrite(name, face)
        print("image written")

	# cut out face from the frame and display image.
        cv2.imshow('face', face)

    if ret == True:

        # Display the resulting frame
        cv2.imshow('Frame',image_resized)
        # cv2.imshow('Frame',frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break	
	
# When everything done, release the capture
cap.release()

# Close all the frames
cv2.destroyAllWindows()
