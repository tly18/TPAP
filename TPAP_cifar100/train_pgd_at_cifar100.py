import torch.optim as optim
import torch.utils.data
from torchvision.datasets import mnist, CIFAR10, CIFAR100
import torchvision
from torchvision import models
from torchvision import transforms as transforms
import torch.nn as nn

from res import ResNet18
from vgg_cifar100 import VGG16

import numpy as np
import os
from advertorch.attacks import LinfPGDAttack,CarliniWagnerL2Attack,DDNL2Attack,SpatialTransformAttack
import torchattacks
from torch.autograd import Variable
import torch.nn.functional as F
import torchattacks
import random
import sys
sys.path.append("..")
from models.WideResNet import WideResNet


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

def calculate_mart_loss(kl, train_batch_size, output_adv, output_nat):

    adv_probs = F.softmax(output_adv, dim=1)
    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == target, tmp1[:, -2], tmp1[:, -1])
    loss_adv = F.cross_entropy(output_adv, target) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

    nat_probs = F.softmax(output_nat, dim=1)
    true_probs = torch.gather(nat_probs, 1, (target.unsqueeze(1)).long()).squeeze()
    loss_robust = (1.0 / train_batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs) *\
        (0.0000001 + output_nat.data.max(1)[1] != target.data).float())
    loss = loss_adv + float(6) * loss_robust
    return loss

def calculate_trades_loss(criterion_kl, train_batch_size, output_adv, output_nat, loss_nat, beta=6.0):
    loss_robust = (1.0 / train_batch_size) * criterion_kl(F.log_softmax(output_adv, dim=1),
                                                    F.softmax(output_nat, dim=1))
    loss = loss_nat + beta * loss_robust
    return loss


if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    seed = 1
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现。
    torch.backends.cudnn.benchmark = True

    # CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    # CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    # normalize = transforms.Normalize(CIFAR100_TRAIN_MEAN,
    #                                  CIFAR100_TRAIN_STD)

    # train_dataset&train_dataset
    train_batch_size = 64   # wrn 64 others 128
    test_batch_size = 64    # wrn 64 others 128
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # normalize,
    ])
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    # normalize,
    ])
    train_set = CIFAR100(root='./dataset', train=True, download=False, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=train_batch_size, shuffle=True, num_workers=1)    # wrn 1 others 10
    test_set = CIFAR100(root='./dataset', train=False, download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=test_batch_size, shuffle=False, num_workers=1)


    #对抗训练
    # model = ResNet18().to(device)
    # model = VGG16().to(device)
    model = WideResNet(depth=34, num_classes=100, widen_factor=10, dropRate=0.0).to(device)

    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3)    #
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 90, 110], gamma=0.1)
    criterion = nn.CrossEntropyLoss().to(device)
    kl = nn.KLDivLoss(reduction='none').to(device)      #reduction='sum' size_average=False batchmean
    criterion_kl = nn.KLDivLoss(size_average=False).to(device) 
  
    epoch = 120

    pth_save_dir = os.path.join('./para/pgd_AT_resnet18_cifar100/wrn/mart/')    #  sat mart trades 
    if not os.path.exists(pth_save_dir):
        os.makedirs(pth_save_dir)
 
    for _epoch in range(epoch):
        scheduler.step(_epoch)
        # 训练
        print("train:")
        model.train()
        train_loss = 0
        train_adv_loss = 0
        train_nat_correct = 0
   
        train_adv_correct = 0
        train_total = 0
        # 一次epoch
        for batch_num, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()  # 梯度归零
           
            model.eval()
            data_adv = generation_adv(model,data,target,step_size=0.007,epsilon=0.031,perturb_steps=10,distance='l_inf')
            data_adv.to(device)
            
            model.train()
            
            optimizer.zero_grad()  # 梯度归零   
            
            output_nat = model(data.float())      # 预测输出
            output_adv = model(data_adv.float())  # 预测输出
            
            loss_nat = criterion(output_nat, target)  # 交叉熵损失
            loss_adv = criterion(output_adv, target)  # 交叉熵损失

            loss = calculate_mart_loss(kl, train_batch_size, output_adv, output_nat)
            # loss = calculate_trades_loss(criterion_kl, train_batch_size, output_adv, output_nat, loss_nat, beta=6.0)

            # (loss_adv).backward()  # 梯度反传 
            loss.backward()  # 梯度反传
            optimizer.step()  # 执行一次优化步骤，通过梯度下降法来更新参数的值

            train_adv_loss += loss_adv.item() # 累加损失值
            prediction_adv = torch.max(output_adv, 1)
            train_adv_correct += torch.sum(prediction_adv[1] == target)

            train_loss += loss_nat.item()  # 累加损失值
            prediction = torch.max(output_nat, 1)  # dim=1指行   
            train_nat_correct += torch.sum(prediction[1] == target)    #正确加1

            train_total += target.size(0) 

        print(_epoch, 'train nat loss', train_loss, ' correct', train_nat_correct / train_total)
        print(_epoch, 'train adv loss', train_adv_loss, ' correct', train_adv_correct / train_total)
 
    
        # 测试
        print("test:")
        model.eval()
        test_loss = 0
        test_nat_correct = 0

        test_adv_loss = 0
        test_adv_correct = 0
        test_total = 0

        # with torch.no_grad():
        for batch_num, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data.float())  # 预测输出
            prediction = torch.max(output, 1)
            loss_ = criterion(output, target)  # 交叉熵损失
            test_loss += loss_.item()
            test_nat_correct += torch.sum(prediction[1] == target)  # 正确次数

            optimizer.zero_grad()  # 梯度归零   
            # 生成对抗样本
            data_adv = pgd_whitebox_attack(model,data,target,epsilon=8/255,num_steps=20,step_size=0.003)
            data_adv.to(device)
            
            output_adv = model(data_adv.float())  # 预测输出
            _loss = criterion(output_adv, target)  # 交叉熵损失
            test_adv_loss += _loss.item()  # 累加损失值
            prediction_adv = torch.max(output_adv, 1)
            test_adv_correct += torch.sum(prediction_adv[1] == target)
            
            test_total += target.size(0)

        print(_epoch, ' loss', test_loss, ' correct', test_nat_correct / test_total)
        print(_epoch, ' test adv loss', test_adv_loss, ' correct', test_adv_correct / test_total)

        if _epoch >= 60:
            torch.save(model.state_dict(), os.path.join(pth_save_dir, '%d_pgd_AT_cifar100.pt' % _epoch))
            # data是前面运行出的数据，先将其转为字符串才能写入
        result2txt = str(_epoch) + ',训练:' + str(round(((train_nat_correct / train_total)*100).item(),4)) + ',' +\
            str(round(((train_adv_correct / train_total)*100).item(),4)) + ',测试:' +\
            str(round(((test_nat_correct / test_total)*100).item(),4)) + ',' +\
            str(round(((test_adv_correct / test_total)*100).item(),4))# data是前面运行出的数据，先将其转为字符串才能写入
    
        with open(pth_save_dir + 'pgd_AT结果存放.txt', 'a') as file_handle:  # .txt可以不自己新建,代码会自动新建
            file_handle.write(result2txt)  # 写入
            file_handle.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
