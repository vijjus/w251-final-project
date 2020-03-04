# w251-final-project

## Smart Doorbell ##

* Leverage data streams coming off a smart doorbell like the ring
* Upload images for parts of the video when there is motion 
* Run it through a model that recognizes people in your family in images and possibly classify the object that is moving (ie cat vs car vs delivery guy)
* Only alert the homeowner when there is motion they care about and possibly classify them. 
 
We plan to leverage a large part of HW3. We can optionally replace opencv Haar cascade filters with a pre-trained CNN such as mtcnn.

Once the image is classified, we will put it in the pipeline for post processing. 

Task 1: Person re-identification task (https://github.com/KaiyangZhou/deep-person-reid). Here, the idea is that our image stream should be able to identify a set of known faces.
 
Task 2: We will also use person re-identification to setup an alarm trigger if a person not known to us is detected by our stream for 30+ seconds.

The model listed above is in PyTorch.

| # | Task                                                                            | Person | End Date | Status |
|---|---------------------------------------------------------------------------------|--------|----------|--------|
| 1 | Re-implement HW3 to setup a baseline                                            |        |          |        |
| 2 | Modify containers to be able to use PyTorch                                     |Patrick |          |        |
| 3 | Replace opencv with mtcnn and get HW3 parity                                    |Vijay/Kat|          |        |
| 4 | Collect dataset of known faces                                                  |        |          |        |
| 5 | Replay collected mp4 data through model so that faces are cropped and captured  |        |          |        |
| 6 | Implement torchreid to identify people in the backend container (static images) |        |          |        |
| 7 | Get #6 to work with streaming data                                              |        |          |        |
| 8 | Implement “linger” detector based on #7                                         |        |          |        |
