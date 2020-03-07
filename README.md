# w251-final-project

## Smart Doorbell ##

* Leverage data streams coming off a smart doorbell like the ring
* Upload images for parts of the video when there is motion 
* Run it through a model that recognizes people in your family in images and possibly classify the object that is moving (ie cat vs car vs delivery guy)
* Only alert the homeowner when there is motion they care about and possibly classify them. 
 
We plan to leverage a large part of HW3. We can optionally replace opencv Haar cascade filters with a pre-trained CNN such as mtcnn.

Once the image is classified, we will put it in the pipeline for post processing. 

### Person Re-Identification ###

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

* Figure out if we want to include it in our project
* Get the container bindings figured out
* Run in Jupyter to experiment with sample images

### Face Detection ###

We could also implement something simpler to start with, which is face detection. There are some pre-trained models available, such as VGGFace and OpenFace. We will use these, and finetune it with a dataset composed of our faces.

We use the following as a reference:

https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78


| # | Task                                                                            | Person | End Date | Status |
|---|---------------------------------------------------------------------------------|--------|----------|--------|
| 1 | Re-implement HW3 to setup a baseline                                            | Vijay/Kat    |          |        |
| 2 | Modify containers to be able to use PyTorch                                     |Patrick |          |        |
| 3 | Replace opencv with mtcnn and get HW3 parity                                    |Vijay/Kat|          |        |
| 4 | Collect dataset of known faces                                                  |All     |          |        |
| 5 | Replay collected mp4 data through model so that faces are cropped and captured  | Vijay      |          |        |
| 6 | Implement torchreid to identify people in the backend container (static images) | Patrick       |          |        |
| 7 | Get #6 to work with streaming data                                              |Vijay/Kat      |          |        |
| 8 | Implement “linger” detector based on #7                                         |        |          |        |
