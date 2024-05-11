# TPAP
import torch.optim as optim
import torch.utils.data
from torchvision.datasets import mnist, CIFAR10
import torchvision
from torchvision import models
import torch.nn as nn
import sys
sys.path.append("..")
from res_tiny import ResNet18
from vgg_tiny import VGG16
from models.WideResNet import WideResNet
import numpy as np
import os
from advertorch.attacks import LinfPGDAttack,CarliniWagnerL2Attack,DDNL2Attack,SpatialTransformAttack
import torchattacks
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import math
import torch.utils.model_zoo as model_zoo
from PIL import Image
import matplotlib.pyplot as plt
import random
import torch, os
import torchvision.datasets as datasets
# import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models

# generate adversarial example
def generation_adv(model,
              x_natural,
              y,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              distance='l_inf'):                                                                                                                                     
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    return x_adv

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

if __name__ == '__main__':
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

    data_dir = './tiny-imagenet-200/'
    num_workers = {'train': 1,'val': 1,'test': 1}

    # channel_means = (0.48043839, 0.44820218, 0.39760034)
    # channel_stdevs = (0.27698959, 0.26908774, 0.28216029)
    # mean = torch.tensor([0.48024578664982126, 0.44807218089384643, 0.3975477478649648]).cuda()
    # std = torch.tensor([0.2769864069088257, 0.26906448510256, 0.282081906210584]).cuda()
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomCrop(64, padding=4),
            # transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            # transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
                    for x in ['train', 'val','test']}
    # dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=num_workers[x])
    #                 for x in ['train', 'val', 'test']}
    
    train_loader = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=num_workers[x])
                for x in ['train']}
    test_loader = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=False, num_workers=num_workers[x])
                for x in ['val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}


    # 加载模型

    # model = ResNet18().to(device)
 
    model_name = str('vgg16') 
    pattern = str('sat')    # mart trades 
    model = VGG16().to(device)
    # model = WideResNet(depth=34, num_classes=200).to(device)   

    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 90, 190], gamma=0.1)
    criterion = nn.CrossEntropyLoss().to(device)

    epoch = 100
    
    adversary = torchattacks.FGSM(model, eps=8/255)
    process = torchattacks.FGSM(model, eps=8/255)

    pth_save_dir = os.path.join('./para/fgsm_AT_tiny_imagenet/'+ model_name + '/' + pattern +'/')
    if not os.path.exists(pth_save_dir):
        os.makedirs(pth_save_dir)
 
    for _epoch in range(epoch):      
        scheduler.step(_epoch)
        # 训练
        print("train:")
        model.train()
        # 计算损失值
        train_nat_loss = 0
        train_adv_one_loss = 0
        train_adv_two_loss = 0
        # 正确个数
        train_nat_correct = 0
        train_adv_one_correct = 0
        train_adv_two_correct = 0
        # 总个数
        train_total = 0
        
        for i,(data, target) in enumerate(train_loader['train']):
            data = data.float().to(device)
            target = target.long().to(device)

            optimizer.zero_grad()  # 梯度归零
           
            model.eval()

            data_adv_one = adversary(data, target)                  #8/255

            model.train()
            optimizer.zero_grad()  # 梯度归零

            data_adv_one.to(device)
            
            output_nat = model(data.float())      # 预测输出
            output_adv_one = model(data_adv_one.float())  # 预测输出
            
            loss_nat = criterion(output_nat, target)  # 交叉熵损失
            loss_adv_one = criterion(output_adv_one, target)  # 交叉熵损失

            loss_adv_one.backward() 
            optimizer.step()  # 执行一次优化步骤，通过梯度下降法来更新参数的值

            #nat
            train_nat_loss += loss_nat.item() # 累加损失值
            prediction_nat = output_nat.max(1)[1]
            train_nat_correct += torch.sum(prediction_nat==target)
            #adv
            train_adv_one_loss += loss_adv_one.item() # 累加损失值
            prediction_adv_one = output_adv_one.max(1)[1]
            train_adv_one_correct += torch.sum(prediction_adv_one==target)

            train_total += target.size(0) 

        print(_epoch, 'train nat loss', train_nat_loss, ' accuracy:', train_nat_correct / train_total)
        print(_epoch, 'train adv loss', train_adv_one_loss, ' accuracy:', train_adv_one_correct / train_total)
 
        # 测试
        print("test:")
        model.eval()
        test_loss = 0
        test_nat_correct = 0
        test_adv_loss = 0
        test_adv_correct = 0
        test_total = 0
        test_adv_p_loss = 0
        test_adv_p_correct = 0

        # with torch.no_grad():
        for batch_num, (data, target) in enumerate(test_loader['val']):

            data, target = data.float().to(device), target.long().to(device)  

            # TPAP:输入是干净样本
            output = model(process(data, torch.max(model(data.float()), 1)[1]).to(device).float())  # 预测输出 
            prediction = torch.max(output, 1)
            loss_ = criterion(output, target)  # 交叉熵损失
            test_loss += loss_.item()
            test_nat_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())  # 正确次数

            optimizer.zero_grad()  # 梯度归零   
            
            # 生成PGD对抗样本
            data_adv = pgd_whitebox_attack(model,data,target,epsilon=0.031,num_steps=20,step_size=0.003)
            data_adv.to(device)
            optimizer.zero_grad()  # 梯度归零 

            output_adv = model(data_adv.float())  # 预测输出
            _loss = criterion(output_adv, target)  # 交叉熵损失
            test_adv_loss += _loss.item()  # 累加损失值
            prediction_adv = torch.max(output_adv, 1)
            test_adv_correct += np.sum(prediction_adv[1].cpu().numpy() == target.cpu().numpy())
                        
            # TPAP: 输入是PGD对抗样本
            data_fgsm_p = process(data_adv, torch.max(model(data_adv.float()), 1)[1])         
            data_fgsm_p.to(device)
            output_adv_p = model(data_fgsm_p.float())  # 预测输出

            p_loss = criterion(output_adv_p, target)  # 交叉熵损失
            test_adv_p_loss += p_loss.item()  # 累加损失值
            prediction_adv_p = torch.max(output_adv_p, 1)
            test_adv_p_correct += np.sum(prediction_adv_p[1].cpu().numpy() == target.cpu().numpy())

            test_total += target.size(0)

        print(_epoch, 'train nat loss', test_loss, ' accuracy:', test_nat_correct / test_total)
        print(_epoch, ' test adv loss', test_adv_loss, ' accuracy:', test_adv_correct / test_total)
        print(_epoch, ' test adv_p loss', test_adv_p_loss, ' accuracy:', test_adv_p_correct / test_total)

        # if _epoch >= 80:
        torch.save(model.state_dict(), os.path.join(pth_save_dir, '%d_FGSM_AT_tiny_%s.pt' % (_epoch, model_name)))
        # data是前面运行出的数据，先将其转为字符串才能写入
        result2txt = str(_epoch) + ',训练:' + str(round(((train_nat_correct / train_total)*100).item(),4)) + ',' +\
            str(round(((train_adv_one_correct / train_total)*100).item(),4)) + ',测试:' +\
            str(round(((test_nat_correct / test_total)*100).item(),4)) + ',' +\
            str(round(((test_adv_correct / test_total)*100).item(),4))+ ',' +\
            str(round(((test_adv_p_correct / test_total)*100).item(),4))+',训练损失:' +\
            str(round(train_nat_loss,4)) + ',' + str(round(train_adv_one_loss,4)) +\
            ',测试损失:' + str(round(test_loss,4)) + ',' + str(round(test_adv_loss,4)) + ',' + str(round(test_adv_p_loss,4))
    
        with open(pth_save_dir + 'AT_'+model_name+'结果存放.txt', 'a') as file_handle:  # .txt可以不自己新建,代码会自动新建
            file_handle.write(result2txt)  # 写入
            file_handle.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
