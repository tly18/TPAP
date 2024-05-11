# 测试其它攻击方法/Test other attack methods

import torch.nn.functional
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision.datasets import mnist, CIFAR10
import torchvision
from torchvision import transforms as transforms
import torch.nn as nn
from models.my_model import Changed_ResNet
import numpy as np
import os
from advertorch.attacks import LinfPGDAttack,CarliniWagnerL2Attack,DDNL2Attack,SpatialTransformAttack,GradientSignAttack
from autoattack import AutoAttack
from models.VGG import VGG16
from models.WideResNet import WideResNet
from models.res import ResNet18
from torchvision import models
import models.zhou as zhou
import models.res as res
from PGD_attack import PGD
from attack.DIM.DIM import DIM_Attack
from FWA.frank_wolfe import FrankWolfe
import sys
import torchattacks
import foolbox
import matplotlib.pyplot as plt
from torch.autograd import Variable
torch.set_printoptions(profile="full")

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def pgd_whitebox_attack(model,
                  X,
                  y,
                  epsilon=0.031,
                  num_steps=20,
                  step_size=0.003):

    X_pgd = Variable(X.data, requires_grad=True)
    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    return X_pgd

def calcu(model, test_loader, adversary, process, i, a, batch_size):
    # 测试
    print("test:")
    model.eval()

    test_loss = 0
    test_correct = 0
    total = 0
    test_adv_loss = 0
    test_adv_correct = 0
    test_adv_total = 0
    test_adv_correct_ = 0
    test_adv_total_ = 0
    test_adv_correct_re = 0
    test_adv_total_re = 0
    
    temp_nat = 0
    temp_adv = 0
    temp_nat_loss = 0
    temp_adv_loss = 0

 
    for batch_num, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)

        # 生成对抗样本
        if i==6:        # aa
            data_adv = adversary.run_standard_evaluation(data.float(), target, bs=batch_size).to(device)   #adv_untargeted
        elif i==8:      # dim
            data_adv = adversary(data.float(), target).to(device)  
        elif i==9:
            data_adv = data.to(device)
        else:
            data_adv = adversary.perturb(data.float(), target).to(device)

        # 测试对抗样本的攻击正确率
        adv_output = model(data_adv.float())  
        prediction_adv_re = torch.max(adv_output, 1)
        test_adv_total_re += target.size(0)
        test_adv_correct_re += np.sum(prediction_adv_re[1].cpu().numpy() == target.cpu().numpy())

        # 对抗性净化
        data_adv = process(data_adv, torch.max(model(data_adv.float()), 1)[1]).to(device)
        
        # 分类结果
        output_adv = model(data_adv.float())  # 测试对抗图像
        loss_tadv = criterion(output_adv, target)  # 交叉熵损失
        test_adv_loss += loss_tadv.item()  # 累加损失值
        prediction_adv = torch.max(output_adv, 1)  # second param "1" represents the dimension to be reduced
        test_adv_total += target.size(0)
        test_adv_correct += np.sum(prediction_adv[1].cpu().numpy() == target.cpu().numpy())


    temp_adv_loss = test_adv_loss
    temp_adv = test_adv_correct / test_adv_total
    print('对抗样本攻击成功率：', 1-(test_adv_correct_re/test_adv_total_re))
    print('总的损失值：', temp_adv_loss, ' 净化样本分类正确率：', temp_adv)
    print('\n')

    # 0：攻击方法的类型 1：对抗样本攻击成功率 2：净化样本分类正确率 3：损失值
    a[i][0] = i
    a[i][1] = 1-(test_adv_correct_re/test_adv_total_re)
    a[i][2] = temp_adv
    a[i][3] = temp_adv_loss

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"]="1"
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # torch.cuda.manual_seed(1)
    # torch.backends.cudnn.benchmark = True
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # CIFAR10 train_dataset&train_dataset
    batch_size = 128   # 由于显卡内存有限，wideresnet34设置为64，其他网络128
    # train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    train_transform = transforms.Compose([transforms.ToTensor()])
    #transforms.RandomRotation((-10,10)),transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip()
    # train_set = CIFAR10(root='./cifar10', train=True, download=False, transform=train_transform)
    # train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False)
    test_set = CIFAR10(root='./cifar10', train=False, download=False, transform=train_transform)
    # print(vars(test_set))
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=10)

    # 加载训练好的模型
    # VGG16
    # model = VGG16().to(device)
    
    # WideResNet
    # model = WideResNet(depth=34, num_classes=10, widen_factor=10, dropRate=0.0).to(device) 

    # ResNet18
    model = res.ResNet18().to(device)
    model.load_state_dict(torch.load('./para/AT_resnet18_fgsm/resnet18/sta_at/99_standard_AT_cifar10_resnet18.pt', map_location=device))
    
    criterion = nn.CrossEntropyLoss(reduction="mean").to(device)
    
    # 净化强度
    process = torchattacks.FGSM(model, eps=8/255)
    
    # 记录输出
    a = np.zeros((10,4))

    for i in [0,10,2]:
        # generate adversarial examples
        if i==0:
            # STA
            print("STA")
            adversary = SpatialTransformAttack(
                model, 10, clip_min=0.0, clip_max=1.0, max_iterations=10, search_steps=5, targeted=False)
            
        if i==1:
            # FGSM
            print("FGSM")
            adversary = GradientSignAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                                                    eps=8 / 255, clip_min=0.0, clip_max=1.0, targeted=False)
            
        if i==2:
            # PGD
            print("PGD")
            adversary = LinfPGDAttack(
                model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8/255,
                nb_iter=20, eps_iter=0.003, rand_init=True, clip_min=0.0,
                clip_max=1.0, targeted=False)
                    
        if i==3:
            # PGD100
            print("PGD100")
            adversary = LinfPGDAttack(
                model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8/255,
                nb_iter=100, eps_iter=0.0007, rand_init=True, clip_min=0.0,
                clip_max=1.0, targeted=False)
            
        if i==4:
            # CW
            print("CW")
            adversary = CarliniWagnerL2Attack(
                model, 10, clip_min=0.0, clip_max=1.0, max_iterations=10, confidence=1, initial_const=1, learning_rate=1e-2,
                binary_search_steps=4, targeted=False)      #max_iterations=100
                        
        if i==5:
            # DDN
            print("DDN")
            adversary = DDNL2Attack(model, nb_iter=20, gamma=0.05, init_norm=1.0, quantize=True, levels=16, clip_min=0.0,
                        clip_max=1.0, targeted=False, loss_fn=None)       #nb_iter=40 levels=256

            
        if i==6:
            # AA
            print("AA")
            adversary = AutoAttack(model, norm='Linf', eps=8/255, device=device, verbose = False)       #verbose=True

        if i==7:
            # FWA
            print("FWA")
            adversary = FrankWolfe(predict=model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                    eps=8/255, kernel_size=4, entrp_gamma=0.003, nb_iter=20, dual_max_iter=15, grad_tol=1e-4,
                    int_tol=1e-4, device=device, postprocess=False, verbose=False)       # verbose=True

        if i==8:
            # DIM
            print("DIM")
            adversary = DIM_Attack(model,
                            decay_factor=1, prob=0.5,
                            epsilon=8/255, steps=20, step_size=0.003,
                            image_resize=33,
                            random_start=False, device=device)
            
        if i==9:
            # clean
            print("clean")
            adversary = None
            
        calcu(model, test_loader, adversary, process, i, a, batch_size)

    print(a)
    print("CIFAR-10, Resnet-18")
        # torchattacks
        # adversary = torchattacks.DeepFool(model, steps=10, overshoot=0.01)
        # adversary = torchattacks.CW(model, c=1, kappa=1, steps=10, lr=0.01)
        # adversary = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=20, random_start=True)
        # adversary = torchattacks.BIM(model, eps=8/255, alpha=2/255, steps=10)

        # trades
        # data_adv = pgd_whitebox_attack(model,data,target,epsilon=0.031,num_steps=20,step_size=0.003)
