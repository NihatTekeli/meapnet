# Margin-Enhanced Average Precision Optimization for Visible-Infrared Person Re-identification
## Overview
This repository contains Pytorch code for the implementation of the methods in the paper "Margin-Enhanced Average Precision Optimization for Visible-Infrared Person Re-identification". The paper introduces MEAP and HSA techniques. 

## Methods
### Margin-Enhanced Average Precision
MEAP is a novel optimization strategy that enhances the average precision based loss function with a margin-based mechanism. This helps to improve the discriminative power of the model in visible-infrared person re-identification scenarios.

Typical feature similarity distribution by Smooth-AP: <br>
<img src="https://github.com/user-attachments/assets/16b5d8fd-c2b4-4ff6-8d85-38b469adf4d0" width=750>

Targeted feature similarity distribution of MEAP: <br>
<img src="https://github.com/user-attachments/assets/d4e81bce-1cc8-4c49-8330-8cbe25fbf917" width=750>

## Horizontal Stripe Augmentation 
Horizontal Stripe Augmentation is a data augmentation technique for visible-infrared person re-identification. This technique generates augmented samples with horizontal stripes from visible and infrared modalities to improve the robustness of the model.

Visible image augmentation: <br>
<img src="https://github.com/user-attachments/assets/c4b1d3a7-1932-4252-91ad-d567761dcc92" width=500>

Infrared image augmentation: <br>
<img src="https://github.com/user-attachments/assets/63c1c0ec-99aa-49ad-b720-0f68cf04f3d1" width=500>

Example augmented images: <br>
<img src="https://github.com/user-attachments/assets/425e0ef2-b2ff-4c60-ad98-a5967da6d310">

## Architecture
The source code is based on the baseline implementation of the AGW method in [1]. The loss function is implemented by modifying the Smooth-AP method in [2]. 

Architecture illustration: <br>
<img src="https://github.com/user-attachments/assets/258887b9-cdf3-41e7-8276-9011cda36113">

## Requirements
Python>=3.10

PyTorch>=1.12

## Dataset Preparation
The RegDB [3] and SYSU-MM01 [4] datasets are used for the experiments. Download the datasets into the 'Datasets' folder in the parent directory. The 'pre_process_sysu.py' script should be run for SYSU-MM01 dataset.

## Training
```
python train.py --dataset regdb --batch-size 6 --meap 1 --hsa 1 --num_parts 6
python train.py --dataset sysu --batch-size 4 --meap 1 --hsa 1 --num_parts 6
```

## Citation
Please kindly cite
```
@article{tekeli2024meapnet,
title = {Margin-enhanced average precision optimization for visible-infrared person re-identification},
author = {Nihat Tekeli and Ahmet Burak Can},
journal = {Computers and Electrical Engineering},
volume = {120},
pages = {109751},
year = {2024},
issn = {0045-7906},
doi = {https://doi.org/10.1016/j.compeleceng.2024.109751},
url = {https://www.sciencedirect.com/science/article/pii/S0045790624006785}
}
```

## References
[1] Ye M, Shen J, Lin G, Xiang T, Shao L, Hoi SC. Deep learning for person re-identification: A survey and outlook. IEEE Trans Pattern Anal Mach Intell
2021;44(6):2872–93.

Source code link: https://github.com/mangye16/Cross-Modal-Re-ID-baseline

[2] Brown A, Xie W, Kalogeiton V, Zisserman A. Smooth-AP: Smoothing the path towards large-scale image retrieval. In: European conference on computer
vision. Springer; 2020, p. 677–94.

Source code link: https://www.robots.ox.ac.uk/~vgg/research/smooth-ap/

[3] Nguyen DT, Hong HG, Kim KW, Park KR. Person recognition system based on a combination of body images from visible light and thermal cameras.
Sensors 2017;17(3):605.

Dataset download link: http://dm.dongguk.edu/link.html (Requires form submission for DBPerson-Recog-DB1)

[4] Wu A, Zheng W-S, Yu H-X, Gong S, Lai J. RGB-infrared cross-modality person re-identification. In: Proceedings of the IEEE international conference on
computer vision. 2017, p. 5380–9.

Dataset download link: https://github.com/wuancong/SYSU-MM01 (Requires form submission)
