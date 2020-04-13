# Image Captioning ##

Replicated the results from:

https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

Fine tuned a ResNet 101 model on the MSCOCO dataset, using captioning information provided by Karpathy.

### Package Requirements ###

* pip3
* torch (version 1.4.0)
* jupyter (for EDA/EMA)
* matplotlib (image rendering)
* OpenCV (for image capture from video)
* torchvision (for pretrained vision models)
* h5py (store image data efficiently)
* tqdm (monitor progress)
* nltk (bleu score metric)
* Nvidia Apex (mixed precision training, FP16)

### Setting up the Data ###

Download the MSCOCO training and validation data:

* Training data: wget http://images.cocodataset.org/zips/train2014.zip (13.5G zipped)
* Validation data: wget http://images.cocodataset.org/zips/val2014.zip (6.6G zipped)
* Andrej Karpathy's captioning data: wget http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip (37M zipped)

In our case, the data was downloaded to: /data/caption_data.

```
root@project:/data/w251-final-project/captioning# cd /data/caption_data/
root@project:/data/caption_data# ls -lrt
total 222948
-rw-rw-r-- 1 root root   9035673 Nov 24  2014 dataset_flickr8k.json
-rw-rw-r-- 1 root root  38318553 Nov 25  2014 dataset_flickr30k.json
-rw-rw-r-- 1 root root 144186139 Nov 26  2014 dataset_coco.json
-rw-r--r-- 1 root root  36745453 Apr 15  2015 caption_datasets.zip
drwxr-xr-x 4 root root      4096 Apr  4 19:41 mscoco

root@project:/data/caption_data/mscoco/train2014# ls -lrt | head
total 13341920
-rw-rw-r-- 1 root root  184515 Aug 16  2014 COCO_train2014_000000447150.jpg
-rw-rw-r-- 1 root root  142237 Aug 16  2014 COCO_train2014_000000353086.jpg
-rw-rw-r-- 1 root root  222661 Aug 16  2014 COCO_train2014_000000330573.jpg
-rw-rw-r-- 1 root root  110714 Aug 16  2014 COCO_train2014_000000291797.jpg
-rw-rw-r-- 1 root root  128572 Aug 16  2014 COCO_train2014_000000165046.jpg
-rw-rw-r-- 1 root root  198413 Aug 16  2014 COCO_train2014_000000114353.jpg
-rw-rw-r-- 1 root root  176758 Aug 16  2014 COCO_train2014_000000113521.jpg
-rw-rw-r-- 1 root root  251856 Aug 16  2014 COCO_train2014_000000530896.jpg
-rw-rw-r-- 1 root root   34689 Aug 16  2014 COCO_train2014_000000524802.jpg
```

The first step is to combine the images from MSCOCO and the captioning information in the **dataset_coco.json** file.

```
root@project:/data/w251-final-project/captioning# python3 create_input_files.py 

Reading TRAIN images and captions, storing to file...

  8%|███████████▎                           | 9075/113287 [01:28<16:07, 107.73it/s]
```

This reads the data downloaded and saves the following files –

* An HDF5 file containing images for each split in an I, 3, 256, 256 tensor, where I is the number of images in the split. Pixel values are still in the range [0, 255], and are stored as unsigned 8-bit Ints.
* A JSON file for each split with a list of N_c * I encoded captions, where N_c is the number of captions sampled per image. These captions are in the same order as the images in the HDF5 file. Therefore, the ith caption will correspond to the i // N_cth image.
* A JSON file for each split with a list of N_c * I caption lengths. The ith value is the length of the ith caption, which corresponds to the i // N_cth image.
* A JSON file which contains the word_map, the word-to-index dictionary.

```
root@project:/data/caption_data/mscoco/caption_data# ls -lrt
total 23781408
-rw-r--r-- 1 root root      155872 Apr  4 19:44 WORDMAP_coco_5_cap_per_img_5_min_word_freq.json
-rw-r--r-- 1 root root   101351081 Apr  4 20:03 TRAIN_CAPTIONS_coco_5_cap_per_img_5_min_word_freq.json
-rw-r--r-- 1 root root 22273132544 Apr  4 20:03 TRAIN_IMAGES_coco_5_cap_per_img_5_min_word_freq.hdf5
-rw-r--r-- 1 root root     2261272 Apr  4 20:03 TRAIN_CAPLENS_coco_5_cap_per_img_5_min_word_freq.json
-rw-r--r-- 1 root root     4472584 Apr  4 20:04 VAL_CAPTIONS_coco_5_cap_per_img_5_min_word_freq.json
-rw-r--r-- 1 root root       99829 Apr  4 20:04 VAL_CAPLENS_coco_5_cap_per_img_5_min_word_freq.json
-rw-r--r-- 1 root root   983042048 Apr  4 20:04 VAL_IMAGES_coco_5_cap_per_img_5_min_word_freq.hdf5
-rw-r--r-- 1 root root     4472598 Apr  4 20:05 TEST_CAPTIONS_coco_5_cap_per_img_5_min_word_freq.json
-rw-r--r-- 1 root root   983042048 Apr  4 20:05 TEST_IMAGES_coco_5_cap_per_img_5_min_word_freq.hdf5
-rw-r--r-- 1 root root       99775 Apr  4 20:05 TEST_CAPLENS_coco_5_cap_per_img_5_min_word_freq.json
```

### Changes to the Model ###

We changed the encoder module from a ResNet101 to a ResNeXt101_32x8d model, pretrained on the Imagenet database.
We added Apex mixed precision training for faster training.


In my training, we see something like this:

```
Epoch: [2][17500/17702]	Batch Time 0.208 (0.211)	Data Load Time 0.000 (0.000)	Loss 3.3597 (3.3827)	Top-5 Accuracy 73.224 (73.526)
Epoch: [2][17600/17702]	Batch Time 0.206 (0.211)	Data Load Time 0.000 (0.000)	Loss 3.2255 (3.3828)	Top-5 Accuracy 77.049 (73.525)
Epoch: [2][17700/17702]	Batch Time 0.207 (0.211)	Data Load Time 0.000 (0.000)	Loss 3.1513 (3.3826)	Top-5 Accuracy 74.792 (73.528)
Validation: [0/782]	Batch Time 0.329 (0.329)	Loss 3.1433 (3.1433)	Top-5 Accuracy 76.786 (76.786)	
Validation: [100/782]	Batch Time 0.156 (0.159)	Loss 3.4992 (3.2905)	Top-5 Accuracy 73.262 (74.708)	
Validation: [200/782]	Batch Time 0.172 (0.158)	Loss 3.3651 (3.2903)	Top-5 Accuracy 76.020 (74.749)
```

Captions are reasonable, but not perfect.

### Container ###

```
FROM nvcr.io/nvidia/pytorch:19.09-py3

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /root
COPY models.py .
COPY utils.py .
COPY BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar caption_model.pth.tar
COPY caption_app.py .

root@project:/data/w251-final-project/captioning# nvidia-docker run --rm --privileged -v /tmp/s3bucket/:/tmp/s3bucket/ -v /tmp/s3kat/:/tmp/s3kat/ --ipc=host -it gpucapt bash

root@c63f5552e545:~# python3 caption_app.py 
cuda:0
/opt/conda/lib/python3.6/site-packages/torch/serialization.py:453: SourceChangeWarning: source code of class 'torch.nn.modules.container.Sequential' has changed. you can retrieve the original source code by accessing the object's source attribute or set `torch.nn.Module.dump_patches = True` and use the patch tool to revert the changes.
  warnings.warn(msg, SourceChangeWarning)
/opt/conda/lib/python3.6/site-packages/torch/serialization.py:453: SourceChangeWarning: source code of class 'torchvision.models.resnet.Bottleneck' has changed. you can retrieve the original source code by accessing the object's source attribute or set `torch.nn.Module.dump_patches = True` and use the patch tool to revert the changes.
  warnings.warn(msg, SourceChangeWarning)
Enter the path to the image: /tmp/s3kat/frame_0b88524c-8305-43fb-9af3-fec6c359ca86.jpg
<start> a woman is standing in front of a fire hydrant <end> 
Enter the path to the image: /tmp/s3kat/frame_909d17ed-1404-49e7-9b0d-636a0a8bc154.jpg
<start> a person standing in front of a fire hydrant <end> 
Enter the path to the image: /tmp/s3kat/frame_7582e5b7-f941-48e9-bc6b-cbc2fe9b9368.jpg
<start> a group of people standing around a fire hydrant <end> 
Enter the path to the image: /tmp/s3kat/frame_7582e5b7-f941-48e9-bc6b-cbc2fe9b9368.jpg
<start> a group of people standing around a fire hydrant <end> 
Enter the path to the image: /tmp/s3kat/frame_93150565-fa5a-4cf8-bc1c-ec80b4ff7ed7.jpg
<start> a group of people standing around a fire hydrant <end> 
Enter the path to the image: /tmp/s3kat/frame_dcaca287-739c-4d22-bda1-341511fb8623.jpg
<start> a couple of people standing next to a fire hydrant <end> 
```
