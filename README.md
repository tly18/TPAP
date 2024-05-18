# Robust Overfitting Does Matter: Test-Time Adversarial Purification With FGSM (CVPR 2024)
## Introduction
<br />下面是这篇文章的代码，主要是用 PyTorch 编写的。
<br />Here's the code for the article, written primarily in PyTorch.
## Usage
### Data Preparation

### Environment Configuration
<br />Pytorch 环境包括 python 3.8.13、pytorch 1.7.1、CUDA 9.2、torchvision 0.8.2、advertorch 0.2.3、torchattacks 3.3.0 和 matplotlib 3.2.2。
<br />The pytorch environment includes python 3.8.13, pytorch 1.7.1, CUDA 9.2, torchvision 0.8.2, advertorch 0.2.3, torchattacks 3.3.0, matplotlib 3.2.2.
<br />
<br />下面是在线使用 conda 和 pip 配置环境的简单示例。
<br />The following is a simple example of configuring an environment online using conda and pip.
<br />
<br />linux环境下安装自己的工作环境python+pytorch+torchvision
<br />Install your own working environment python+pytorch+torchvision in linux environment
<br />
<br />**1、创建环境**
<br />**1. Create environment**
<br />在linux终端输入： 
````
conda create -n tly18 python=3.8.13
````
<br />**2、安装部署**
<br />**2. Deployment**
<br />进入pytorch官网 https://pytorch.org/get-started/previous-versions/ 确定安装版本。
<br />Go to the pytorch website https://pytorch.org/get-started/previous-versions/ to determine the version to install.
<br />在linux终端输入： 
````
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=9.2 -c pytorch
````
<br />**3、安装攻击库**
<br />**3. Installation of attack libraries**
<br />在linux终端输入： 
````
pip install torchattacks
pip install advertorch=0.2.3 -i https://pypi.tuna.tsinghua.edu.cn/simple/
````
### Instructions for the use of FWA
<br />参考FWA方法的使用说明，网址链接如下：
<br />Refer to the FWA method for instructions on how to use the URL link below:
````
https://github.com/tly18/TPAP/blob/main/FWA/README.md
````
