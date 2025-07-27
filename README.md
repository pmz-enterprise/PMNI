# PMNI: Pose-free Multi-view Normal Integration for Reflective and Textureless Surface Reconstruction
Project page:  https://pmz-enterprise.github.io/PMNI/

<p align="center">
  <br>
    <a href="http://arxiv.org/abs/2504.08410">
      <img src='https://img.shields.io/badge/arXiv-Paper-981E32?style=for-the-badge&Color=B31B1B' alt='arXiv PDF'>
    </a>

# Correction
Equation (13) in the main text of the paper is incorrect. It should be:

$$\alpha \equiv \frac{\mathbf{z}^{r} \cdot \mathbf{z}^{ni}}{\mathbf{z}^{ni} \cdot \mathbf{z}^{ni}}.$$



# Quick Start
Code was tested on Ubuntu 20.04 using Python 3.8, PyTorch 2.1.0, and CUDA 11.8 on an Nvidia RTX4090 (24GB). 
**Before started, please ensure CUDA is installed in your environment.**
It is required by [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).

<details><summary> You should see something like the following after typing `nvcc --version` </summary>

```commandline
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:33:58_PDT_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0
```
</details>

Clone the repository and prepare the conda environment:
```commandline
git clone https://github.com/pmz-enterprise/PMNI
cd PMNI 
. ./create_env.sh
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" # install pytorch3d
pip install pyexr # install pyexr
```


# Data
Our data is available at :

Google drive:
https://drive.google.com/drive/folders/1lIw3eXFPiL-h9nY5k6LpYKQHvakFE_fq?usp=drive_link


# Train
```commandline
CUDA_VISIBLE_DEVICES=X python exp_runner.py --conf config/xx.conf --obj_name xx
```

# For your own data
We strongly recommend using [DUSt3R](https://github.com/naver/dust3r) to provide a reasonable initial pose.



# Video
Our supplementary video is available at: 
https://youtu.be/hLIZG24m1Wo


# Bibtex
```
@inproceedings{pmni,
title = {PMNI: Pose-free Multi-view Normal Integration for Reflective and Textureless Surface Reconstruction},
author = {Mingzhi, Pei and Xu, Cao and Xiangyi, Wang and Heng, Guo and Zhanyu, Ma },
year = {2025},
booktitle = CVPR,
}
```

# Acknowledgement
Our implementation is built from [SuperNormal](https://github.com/CyberAgentAILab/SuperNormal), [NoPe-NeRF](https://github.com/ActiveVisionLab/nope-nerf), [BARF](https://github.com/chenhsuanlin/bundle-adjusting-NeRF).

