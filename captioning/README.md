# Image Captioning ##

Replicated the results from:

https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

Fine tuned a ResNet 101 model on the MSCOCO dataset, using captioning information provided by Karpathy.

Minor changes needed to the files in the repo.

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
