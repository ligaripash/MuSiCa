
## Knowing When to Quit: Selective Cascaded Regression withPatch Attention for Real-Time Face Alignment

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![](arch-coarse-no-pga.png)


## Introduction
[Knowing When to Quit: Selective Cascaded Regression withPatch Attention for Real-Time Face Alignment](https://arxiv.org/abs/2012.15460)


## Installation
The codebases are built on top of [MDM](https://github.com/trigeorgis/mdm)

#### Steps
##### Run docker:
  1. Download the docker image.
  2. Load the image: nvidia-docker load < kwtc_docker_image.tar.gz
  3. Run the image: nvidia-docker run -v your_download_dir:dest_dir -it kwtc:new /bin/bash (The -v is needed to copy files to your container)

##### git clone:
  4. Inside the container: cd /opt/kwtc/
  5. git clone https://github.com/ligaripash/MuSiCa.git
 
##### WFLW:
  6. Download the WFLW from here.
  7. copy WFLW.tar.gz to /opt/kwtc/
  8. gunzip WFLW.tar.gz
  9. tar xvf WFLW.tar
  
##### Models:
  10. Download the model from here.
  11. Copy models.tar.gz to /opt/kwtc/
  12. gunzip models.tar.gz
  13. tar xvf models.tar
  
##### Run inference on a pretrained model with 49 patches:
  14. cd MuSica
  15. python inference.py ( inference.json contains the inference paramters). The output is written to /opt/kwtc/output/
  16. Render the calculated landmarks on image: python show_flm_on_image.py ( the output images are written to /tmp/ )
  
 ##### Evaluate the inference against WFLW ground-truth (expression subset)
  17. python evaluate.py (evaluate.json contains the evluation parameters). You should get 0.088 average normalized error.
  
  
 ##### To train the model:
  
  18. python train.py (train_params.py contain the trainig parameters)
  
  
