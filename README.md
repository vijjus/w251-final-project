# Smart Doorbell: w251-final-project

## Idea ##

## Data Pipeline ##

* Leverage data streams coming off a smart doorbell like the ring
* Upload images for parts of the video when there is motion 
* Run it through a model that recognizes people in your family in images and possibly classify the object that is moving (ie cat vs car vs delivery guy)
* Only alert the homeowner when there is motion they care about and possibly classify them. 
 
We plan to leverage a large part of HW3. We can optionally replace opencv Haar cascade filters with a pre-trained CNN such as mtcnn.

Once the image is classified, we will put it in the pipeline for post processing. 

* Figure out if we want to include it in our project
* Get the container bindings figured out
* Run in Jupyter to experiment with sample images

## Implementation ##


### Face Detection ###

The first step in our pipeline is face detection.

In this step, we use a video reader such as __cv2__ or __mmcv__ to split the input video into constituent frames. Each frame is then passed through a face detector to check for the presence of people in the image. Initially, we used __Haar cascade classifier__ available as part of OpenCV [2] to detect faces. The Haar classifier is machine learnt model that is build using a large number of micro-features, such as edges and corners, and trained with positive and negative class examples. The OpenCV implementation is trained using __Adaboost__. [XXX: Add info on fps with Haar]. However, the Haar classifier is not a neural network.

Next, we used a CNN based face detector called __MTCNN__ (multi-tasked CNN) [3]. This model uses a cascade of 3 CNNs that are called the Proposal Net (P-net), Refine Net (R-net) and Output Net (O-net). Each CNN is trained using a different loss function - a face/no-face log loss for P-net, bounding box L2 loss for R-net and facial landmark L2 loss for O-net.

![alt text](mtccn1.jpg "MTCNN networks")

![alt text](mtccn2.jpg "MTCNN networks")


### Face Identification ###

### Image Captioning ###

During the project review, we were given feedback about exploring the ability to detect more complicated things, such as a person delivering a package. We will experiment with image captioning pre-trained models for this:

https://www.tensorflow.org/tutorials/text/image_captioning#download_and_prepare_the_ms-coco_dataset

The smart doorbell collects videos and uploads them to the cloud if the on device model indicates that an unknown person is at the door. In our data pipeline, the Facial Detection module receives the video stream, and splits the video stream into frames and uses a face detection model to look for faces in the stream of faces. Once the frames are compared with the library of known faces/people, we have a sequence of frames in the case we have an unrecognized person. An additional piece of ‘smartness’ we want to add to our project is the ability to inform the owner about the activity that the unknown person above is doing. For e.g. the unknown person could be the mailman, and he may be dropping off mail. More interestingly, the unknown person could be a thief and he may be stealing a package that was left at the door. The latter represents a growing problem with the increasing use of online shopping. 

According to an assessment done by security.org, 8 out of 10 American adults are online shoppers, and Americans spend close to $600B on online shopping. The same study reports that 38% of their survey respondents report being victims of package theft. This fact is also reflected empirically in our own neighborhoods. Everyday, platforms such as Nextdoor, people complain about package thefts. In many cases, people have managed to capture images of the perpetrators in the act. However, perps often know where cameras or surveillance equipment is installed, and avoid a direct view of their faces through the use of hoodies, scarves etc.

In light of the above, our solution addresses this issue by attempting to identify the act of theft instead of trying to identify or capture images of the thief. When such an act is detected, a notification is sent in real time to the homeowner, who can take immediate action.

The essence of the activity detection is an image captioning model. The idea of image captioning is simple - it combines the computer vision task of object detection with an NLP text generation model that is trained to label the objects identified in the image. The idea is explained in the paper, Show, Attend and Tell: Neural Image Caption Generation with Visual Attention [1].

The paper describes the construction of the model:

### Additional Investigation: Person Re-Identification ###

Task 1: Person re-identification task (https://github.com/KaiyangZhou/deep-person-reid). Here, the idea is that our image stream should be able to identify a set of known faces.
 
Task 2: We will also use person re-identification to setup an alarm trigger if a person not known to us is detected by our stream for 30+ seconds.

The model listed above is in PyTorch.

https://medium.com/@niruhan/a-practical-guide-to-person-re-identification-using-alignedreid-7683222da644

https://github.com/zlmzju/part_reid/blob/master/demo/demo.ipynb

Note: I found another repo with much better documentation and a seemingly better model

https://github.com/layumi/Person_reID_baseline_pytorch

Here is a good summary of the person re-id task:

"Person re-ID can be viewed as an image retrieval problem. Given one query image in Camera A, we need to find the images of the same person in other Cameras. The key of the person re-ID is to find a discriminative representation of the person."

So, if we have multiple cameras, then given an image from camera A, we want to know which images from cameras B, C, D, ... contain the same person. The initial image is called the query image, and the model outputs a gallery that is a collection of images. This is mostly useful when we cannot do reliable face detection (image is from a distance etc).

I was able to download the market1501 dataset, and train a ResNet50 model and also run a test iteration using the instructions from this site. The next goals would be:

## References ##

[1] https://arxiv.org/pdf/1502.03044.pdf

[2] https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html

[3] https://arxiv.org/abs/1604.02878
