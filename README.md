# LinkLM
## Introduction
This is the repository for [Improving Referring Ability for Biomedical Language Models
](https://openreview.net/pdf?id=v5YGQK1qCP), including the codes for training LinkLM.

## Installation
### Clone the repository
```shell
git clone https://github.com/Coldog2333/BioLinkLM.git
cd med-eval
```

### Create conda environment
```shell
conda create -n biolinklm python=3.9
```

### Preliminary
#### PyTorch
We recommend the following installation command for PyTorch since we only verify our codes with PyTorch 1.13.1 + CUDA 11.7. You can find more information on the [official website](https://pytorch.org/).
```shell
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```
#### Others
```shell
pip install -r requirements.txt

# rollback numpy to 1.X
pip install numpy==1.26.4
```
#### Flash-attention
```shell
pip install flash-attn --no-build-isolation
```
Detailed information for installation of Flash-attention can be found from: https://github.com/Dao-AILab/flash-attention.

