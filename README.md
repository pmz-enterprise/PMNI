# PMNI: Pose-free Multi-view Normal Integration for Reflective and Textureless Surface Reconstruction
Project page:  https://pmz-enterprise.github.io/PMNI/

<p align="center">
  <br>
    <a href="http://arxiv.org/abs/2504.08410">
      <img src='https://img.shields.io/badge/arXiv-Paper-981E32?style=for-the-badge&Color=B31B1B' alt='arXiv PDF'>
    </a>

# Correction: 
Equation (13) in the main text of the paper is incorrect. It should be:
$$\alpha \equiv \frac{\mathbf{z}^{r} \cdot \mathbf{z}^ni}{\mathbf{z}^ni \cdot \mathbf{z}^ni}.$$


# Abstract
Reflective and textureless surfaces remain a challenge in multi-view 3D reconstruction.Both camera pose calibration and shape reconstruction often fail due to insufficient or unreliable cross-view visual features.
To address these issues, we present PMNI (Pose-free Multi-view Normal Integration), a neural surface reconstruction method that incorporates rich geometric information by leveraging surface normal maps instead of RGB images. By enforcing geometric constraints from surface normals and multi-view shape consistency within a neural signed distance function (SDF) optimization framework, PMNI simultaneously recovers accurate camera poses and high-fidelity surface geometry. Experimental results on synthetic and real-world datasets show that our method achieves state-of-the-art performance in the reconstruction of reflective surfaces, even without reliable initial camera poses.

# Data
Our data is available at :

Google drive:
https://drive.google.com/drive/folders/17Z2dsgfqZRzqu-7xMk8Z1QWBATu48zGz?usp=sharing

# Code
I'm adding some config files

# Video
Our supplementary video is available at: 
https://youtu.be/hLIZG24m1Wo


# Bibtex
```
@inproceedings{pmni2025pei,
title = {PMNI: Pose-free Multi-view Normal Integration for Reflective and Textureless Surface Reconstruction},
author = {Mingzhi, Pei and Xu, Cao and Xiangyi, Wang and Heng, Guo and Zhanyu, Ma },
year = {2025},
booktitle = CVPR,
}
```
