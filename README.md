This repository is the official PyTorch implementation of the paper "Indirect attention: IA-DETR for one shot object detection". The paper will be provided once reviewed. 




-------
&nbsp;
## Brief Introduction

IA-DETR is a state-of-the-art one-shot object detector based on DETR and indirect attention. The indirect attention exploits the transformers for correlating the three main elements of object queries, target image, and query image all at once. The figure below shows what an indirect attention is and how it differs from the self-attention and cross-attention.


<div align=center>  
<img src='.assets/indirect-attention.jpg' width="60%">
</div>


<div align=center>  
<img src='.assets/IADETR_architecture.jpg' width="95%">
</div>



-------
&nbsp;

## Installation

### Pre-Requisites
You must have NVIDIA GPUs to run the codes.

The implementation codes are developed and tested with the following environment setups:
- 8x NVIDIA V100 GPUs (32GB)
- CUDA 12.0
- Python == 3.9
- PyTorch == 2.1.0+cu121, TorchVision == 0.16.0+cu121
- cython, pycocotools, tqdm, scipy
- see requirements.txt for more


### Data Preparation

coming soon ..