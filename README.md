# A photometric stereo method


## Environment

Implemented in PyTorch with Ubuntu 18.04.

Python: 3.9 

PyTorch 1.1.0 with scipy, numpy, etc.

RTX 3090 (24G)

## download two training datasets:

## download these test datsets:



## Testing on your device:
```shell
python eval/run_model.py --retrain xxx.pth.tar --in_img_num X 
```
You can change X to adjust the number of the input image. 

## Results on the DiLiGenT benchmark dataset:

We have provided the error maps on the DiLiGenT benchmark dataset (under 96 input images)



