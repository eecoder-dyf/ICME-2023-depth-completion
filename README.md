# A Two-stage hybrid CNN-Transformer Network for RGB Guided Indoor Depth Completion
This repo is the official code of "A Two-stage hybrid CNN-Transformer Network for RGB Guided Indoor Depth Completion"
Published on ICME 2023

## Dependencies
* Pytorch>=1.10 with CUDA>=11.3
* tensorboard
* imageio
* opencv-python

## Dataset
Please prepare your dataset as the following structure:
```
-folder
    -train
        -gt
            -xxx.png
            -xyy.png
            -...
        -raw
            -xxx.png
            -xyy.png
            -...
        -rgb
            -xxx.png
            -xyy.png
            -...
    -val
        -(the same as train)
    -test
        -(the same as train)
```

## Code introduction
* `net.py`: Code of Self Completion Module.
* `net_bak.py`: Code of Guided Completion Module.
* `swin_util.py`: Code of Cross Modal Transformer block.
* `NYUdataset2.py` Code of dataset loader.
* `criteria.py`: Code of Metrics and loss function.
* `unet_swin2_sun/main0_multi.py`: Code of training and testing Self Completion Module.
* `unet_swin2_sun/main2_multi.py`: Code of training and testing Guided Completion Module.
* folder`swin/` is copied from the official repo of [Swin Transformer](https://github.com/microsoft/Swin-Transformer).
## Get Started (Take SunRGBD Dataset as example)
We train our network by a two-stage manner, frist we train the Self Completion Moudle, then train the Guided Completion Module.
* **Training**
```bash
cd unet_swin2_sun
python -W ignore main0_multi.py --save --batch-size 32 -d /home/dyf/database/SunRGBD/SUNRGBD/data/ -lr 1e-4 --epoch 1000 --cuda 0 1
python -W ignore main2_multi.py --save --batch-size 16 -d /home/dyf/database/SunRGBD/SUNRGBD/data/ -lr 1e-4 --epoch 1000 --cuda 0 1
```
* **Testing**
After training SCM and GCM you can test the model by main2_multi.py
```bash
cd unet_swin2_sun
python -W ignore main2_multi.py --save --batch-size 16 -d /home/dyf/database/SunRGBD/SUNRGBD/data/ -lr 1e-4 --epoch 1000 --cuda 0 1 --test
# save pictures
python -W ignore main2_multi.py --save --batch-size 16 -d /home/dyf/database/SunRGBD/SUNRGBD/data/ -lr 1e-4 --epoch 1000 --cuda 0 1 --test --savepic
```