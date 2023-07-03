# A Two-stage hybrid CNN-Transformer Network for RGB Guided Indoor Depth Completion
This repo is the official repository of "[A Two-stage hybrid CNN-Transformer Network for RGB Guided Indoor Depth Completion](https://raw.githubusercontent.com/eecoder-dyf/ICME-2023-depth-completion/blob/main/paper/A_Two-stage_Hybrid_CNN-Transformer_Network_for_RGB_Guided_Indoor_Depth_Completion.pdf)"
Published on ICME 2023

## Dependencies
* Pytorch≥1.10 with CUDA≥11.3
* tensorboard
* imageio
* opencv-python

## Dataset
Please prepare your dataset as the following structure:
```
-folder
    -train
        -gt
            - *.png
        -raw
            - *.png
        -rgb
            - *.png
    -val
        -(the same as train)
    -test
        -(the same as train)
```
* **Dataset download link**

    [NYUv2](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat) and [SunRGBD](https://rgbd.cs.princeton.edu/data/SUNRGBD.zip) \
    Note that we use the 1449 densely labeled pairs of aligned RGB and depth images. For NYUv2, we randomly selected 100 pictures for validation and 100 pictures for test, the rest for training; For SunRGBD, we randomly select 250 for validation and 250 for test, the rest for training.

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
python -W ignore main0_multi.py --save --batch-size 32 -d /dir/of/your/data/folder/ -lr 1e-4 --epoch 1000 --cuda 0 1
python -W ignore main2_multi.py --save --batch-size 16 -d /dir/of/your/data/folder/ -lr 1e-4 --epoch 1000 --cuda 0 1
```
* **Testing**
After training SCM and GCM you can test the model by main2_multi.py
```bash
cd unet_swin2_sun
python -W ignore main2_multi.py --save --batch-size 16 -d /dir/of/your/data/folder/ -lr 1e-4 --epoch 1000 --cuda 0 1 --test
# save pictures
python -W ignore main2_multi.py --save --batch-size 16 -d /dir/of/your/data/folder/ -lr 1e-4 --epoch 1000 --cuda 0 1 --test --savepic
```
* **Pretrained Models**

    We Release our pretrained models in [Google Drive](https://drive.google.com/drive/folders/1QNvR9ROXQPgH7v9KSYGFDIVmcfP9sMkl?usp=sharing) and [百度云](https://pan.baidu.com/s/1o_Qh3s-ysitzfGzjC9bZfA?pwd=icme).  The `checkpoint_best_loss.pth.tar` is for Self Completion Module and `second_checkpoint_best_loss.pth.tar` is for the Guided Completion Module.
    