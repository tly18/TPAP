# Robust Overfitting Does Matter: Test-Time Adversarial Purification With FGSM (CVPR 2024)
## Introduction
<br /> * 下面是这篇文章的代码，主要是用 PyTorch 编写的。
<br /> * Here's the code for the article, written primarily in PyTorch.
## Usage
### Data Preparation
<br />1、加载数据集时，如果数据集不存在，将下述的代码中**download=False**修改为**download=True**。
<br />1. When loading a dataset, if the dataset does not exist, change download=False to download=True in the following code.
````
CIFAR10(root='./cifar10/', train=True, download=False, transform=transform_train)
CIFAR10(root='./cifar10/', train=False, download=False, transform=transform_test)
````
<br />2、使用Tiny-ImageNet数据集前参考下面链接，处理val数据。
<br />2. Refer to the link below before using the Tiny-ImageNet dataset to process the val data.
````
https://github.com/tly18/TPAP/blob/main/TPAP_tiny_imagenet/val_data.py
````
<br />在pytorch环境下的linux终端运行文件：
````
python val_data.py
````

### Environment Configuration
<br />Pytorch 环境包括 python 3.8.13、pytorch 1.7.1、CUDA 9.2、torchvision 0.8.2、advertorch 0.2.3、torchattacks 3.3.0 和 matplotlib 3.2.2。
<br />The pytorch environment includes python 3.8.13, pytorch 1.7.1, CUDA 9.2, torchvision 0.8.2, advertorch 0.2.3, torchattacks 3.3.0, matplotlib 3.2.2.
<br />
<br />下面是在线使用 conda 和 pip 配置环境的简单示例。
<br />The following is a simple example of configuring an environment online using conda and pip.
<br />
<br />linux环境下安装自己的工作环境 python+pytorch+torchvision。
<br />Install your own working environment python+pytorch+torchvision in linux environment.
<br />
<br />**1、创建环境**
<br />**1. Create environment**
<br />在linux终端输入： 
<br />Type in the linux terminal:
````
conda create -n tly18 python=3.8.13
````
<br />**2、安装部署**
<br />**2. Deployment**
<br />进入pytorch官网 https://pytorch.org/get-started/previous-versions/ 确定安装版本。
<br />Go to the pytorch website https://pytorch.org/get-started/previous-versions/ to determine the version to install.
<br />
<br />在linux终端输入：
<br />Type in the linux terminal:
````
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=9.2 -c pytorch
````
<br />**3、安装攻击库**
<br />**3. Installation of attack libraries**
<br />在linux终端输入： 
<br /><br />Type in the linux terminal:
```
pip install torchattacks
```
```
pip install advertorch=0.2.3 -i ``` https://pypi.tuna.tsinghua.edu.cn/simple/


### Instructions for the use of FWA
<br />参考FWA方法的使用说明，网址链接如下：
<br />Refer to the FWA method for instructions on how to use the URL link below:
````
https://github.com/tly18/TPAP/blob/main/FWA/README.md
````

### Train and Test
<br />我们提供了CIFAR-10、CIFAR-100、SVHN、TinyImageNet四个数据集的TPAP训练方法。
<br />We provide TPAP training methods for four datasets, CIFAR-10, CIFAR-100, SVHN, and TinyImageNet.
<br />
<br />除CIFAR-10在./TPAP目录运行外，CIFAR-100、SVHN、TinyImageNet数据集需要进入各自的目录进行训练。
<br />With the exception of CIFAR-10 which is run in the . /TPAP directory to run, the CIFAR-100, SVHN, and TinyImageNet datasets require access to their respective directories for training.
<br />
<br />例如，训练CIFAR-10：
<br />For example, training CIFAR-10:
````
python train_fgsm_at_cifar10_TPAP.py
````
<br />与Trades、MART方法结合时，需要在文件中修改训练的损失函数。
<br />When combined with the Trades, MART method, the loss function for training needs to be modified in the file.
<br />
<br />我们提供了PGD对抗训练的代码，运行：
<br />We provide code for PGD adversarial training that runs:
````
python train_pgd_at_cifar10.py
````
<br />
<br />我们提供了TPAP在其它攻击方法下的测试代码，运行：
<br />We provide test code for TPAP running under other attack methods:
````
python TEST_other_attack.py
````
### Results
<br />参考文章中提供的结果：
<br />Refer to the results provided in the article, article link:
````
https://arxiv.org/abs/2403.11448
````
## Contact
<br />If you have any problem about our code, feel free to contact us.
<br />**Lei Zhang (leizhang@cqu.edu.cn)**
<br />**LinYu Tang (linyutang@cqu.edu.cn or 2367300330@qq.com)**
<br />or describe it in Issues.
## Citation
````
@article{tang2024robust,
  title={Robust Overfitting Does Matter: Test-Time Adversarial Purification With FGSM},
  author={Tang, Linyu and Zhang, Lei},
  journal={arXiv preprint arXiv:2403.11448},
  year={2024}
}
````
