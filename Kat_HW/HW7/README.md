# Homework 7 - Katayoun Borojerdi

### Overview
The objective of this homework is to modify the OpenCV-based face detector processing pipeline implemented in homework 3 and replace it with a Deep Learning-based one.

### Questions

#### Describe your solution in detail. What neural network did you use? What dataset was it trained on? What accuracy does it achieve?

I modified the docker container from tensorrtlab05 to add opencv so I could stream from the webcam attached to the Jetson. The new dockerfile.tensorrthw07 is attached in this folder.

I updated the face_detect python file from hw3 to include the sample neural network from the hw07-hint jupyter notebook. The NN_face_detect python file is attached.

I used the mobilenet SSD(single shot multibox detector) based face detector from the example Notebook. The model was pretrained model by WIDERFACE dataset. 

#### Does it achieve reasonable accuracy in your empirical tests? Would you use this solution to develop a robust, production-grade system?

I found the accuracy to be pretty reasonable. You can set a thresshold value to limit false positives, but this may mean that you miss some frames. I found that the NN-detector did better when the face where far away. For some reason when the face took up the majority of the image it had a harder time detecting the face at the threshold confidance level. It did seem to do a good job on multiple faces as well. I think that this solution could be used, but it might need a bit of refinement.

#### What framerate does this method achieve on the Jetson? Where is the bottleneck?

I estimated the fps by counting the number of frames during a stream and dividing by the elapsed time of the stream. The result was approx 4 fps. 

from reading this: https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/ article.  
It seems that accessing USB camera using the ```cv2.VideoCapture``` function and the ```.read()```  method blocks the our Python script until the frame is read from the camera device, we perform our NN detection, and then return to the script. So it seems that the camera I/O could be the bottleneck.

#### Which is a better quality detector: the OpenCV or yours?

I think the 2 detectors seems to both work well. They seemed to be better at different things. I think the NN-detector might allow for more flexibility in re-training, if you want to add classes, or otherwise change the model to try and speed up detection.
