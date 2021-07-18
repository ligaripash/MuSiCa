
## Knowing When to Quit: Selective Cascaded Regression withPatch Attention for Real-Time Face Alignment

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![](arch-coarse-no-pga.png)


## Introduction
[Knowing When to Quit: Selective Cascaded Regression withPatch Attention for Real-Time Face Alignment](https://arxiv.org/abs/2012.15460)


## 300W

## AFLW

#### Notes
- We observe about 1 MOTA noise.
- If the resulting MOTA of your self-trained model is not desired, playing around with the --track_thresh sometimes gives a better performance.
- The default track_thresh is 0.4, except for 0.5 in crowdhuman.
- The training time is on 8 NVIDIA V100 GPUs with batchsize 16.
- We use the models pre-trained on imagenet.
- (crowdhuman, mot17_half) is first training on crowdhuman, then fine-tuning on mot17_half.




## Installation
The codebases are built on top of [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) and [CenterTrack](https://github.com/xingyizhou/CenterTrack).

#### Requirements
- Linux, CUDA>=9.2, GCC>=5.4
- Python>=3.7
- PyTorch ≥ 1.5 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  You can install them together at [pytorch.org](https://pytorch.org) to make sure of this
- OpenCV is optional and needed by demo and visualization


#### Steps
1. Install and build libs
```
git clone https://github.com/PeizeSun/TransTrack.git
cd TransTrack
cd models/ops
python setup.py build install
cd ../..
pip install -r requirements.txt
```

2. Prepare datasets and annotations
```
mkdir crowdhuman
cp -r /path_to_crowdhuman_dataset/CrowdHuman_train crowdhuman/CrowdHuman_train
cp -r /path_to_crowdhuman_dataset/CrowdHuman_val crowdhuman/CrowdHuman_val
mkdir mot
cp -r /path_to_mot_dataset/train mot/train
cp -r /path_to_mot_dataset/test mot/test
```
CrowdHuman dataset is available in [CrowdHuman](https://www.crowdhuman.org/). 
```
python3 track_tools/convert_crowdhuman_to_coco.py
```
MOT dataset is available in [MOT](https://motchallenge.net/).
```
python3 track_tools/convert_mot_to_coco.py
```

3. Pre-train on crowdhuman
```
sh track_exps/crowdhuman_train.sh
python3 track_tools/crowdhuman_model_to_mot.py
```
The pre-trained model is available [crowdhuman_final.pth](https://drive.google.com/drive/folders/1DjPL8xWoXDASrxgsA3O06EspJRdUXFQ-?usp=sharing).

4. Train TransTrack
```
sh track_exps/crowdhuman_mot_trainhalf.sh
```

5. Evaluate TransTrack
```
sh track_exps/mot_val.sh
sh track_exps/mota.sh
```

6. Visualize TransTrack
```
python3 track_tools/txt2video.py
```


## Test set
Pre-training data | Fine-tuning data | Training time | MOTA% | FP | FN | IDs
:---:|:---:|:---:|:---:|:---:|:---:|:---:
crowdhuman | mot17 | ~40h + 2h | 68.4 | 22137  | 152064  | 3942  
crowdhuman | crowdhuman + mot17 | ~40h + 6h | 74.5 | 28323 | 112137 | 3663 

#### Notes
- Performance on test set is evaluated by [MOT challenge](https://motchallenge.net/).
- (crowdhuman + mot17) is training on mixture of crowdhuman and mot17.
- We won't release trained models for test test. Running as in Steps could reproduce them. 
 

## License

TransTrack is released under MIT License.


## Citing

If you use TransTrack in your research or wish to refer to the baseline results published here, please use the following BibTeX entries:

```BibTeX

@article{transtrack,
  title   =  {TransTrack: Multiple-Object Tracking with Transformer},
  author  =  {Peize Sun and Jinkun Cao and Yi Jiang and Rufeng Zhang and Enze Xie and Zehuan Yuan and Changhu Wang and Ping Luo},
  journal =  {arXiv preprint arXiv: 2012.15460},
  year    =  {2020}
}

```
