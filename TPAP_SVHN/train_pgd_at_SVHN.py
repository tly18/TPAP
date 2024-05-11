# PGD对抗训练
import torch.optim as optim
import torch.utils.data
from torchvision.datasets import mnist, CIFAR10
import torchvision
from torchvision import models
from torchvision import transforms as transforms
import torch.nn as nn
import numpy as np
import os
from advertorch.attacks import LinfPGDAttack,CarliniWagnerL2Attack,DDNL2Attack,SpatialTransformAttack,GradientSignAttack
import torchattacks
from torch.autograd import Variable
import torch.nn.functional as F
import torchattacks
import scipy.io
import random
from res import ResNet18
import sys
sys.path.append("..")
from models.VGG import VGG16
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

    seed = 1
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现。
    torch.backends.cudnn.benchmark = True

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # SVHN train_dataset&train_dataset
    train_batch_size = 128  # 64
    test_batch_size = 128   # 64

    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.SVHN(
        root='./SVHN', split='train', download=False, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=train_batch_size, shuffle=True)

    testset = torchvision.datasets.SVHN(
        root='./SVHN', split='test', download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=False)


    # 加载模型
    model = ResNet18().to(device)
    # model = VGG16().to(device)
    # model = WideResNet(depth=34, num_classes=10, widen_factor=10, dropRate=0.0).to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 90, 110], gamma=0.1)
    criterion = nn.CrossEntropyLoss().to(device)
    kl = nn.KLDivLoss(reduction='none').to(device)     
    criterion_kl = nn.KLDivLoss(size_average=False).to(device) 

    epoch = 120

    pth_save_dir = os.path.join('./para/pgd_AT_SVHN/')

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

        # 正确个数
        train_nat_correct = 0
        train_adv_one_correct = 0

        # 总个数
        train_total = 0

        # 一次epoch
        for batch_num, (data, target) in enumerate(train_loader):
            data, target = data.float().to(device), target.long().to(device)
            
            optimizer.zero_grad()  # 梯度归零
           
            model.eval()
            data_adv = generation_adv(model,data,target,step_size=0.007,epsilon=0.031,perturb_steps=10,distance='l_inf')

            model.train()
            optimizer.zero_grad()  # 梯度归零

            data_adv.to(device) 
            
            output_nat = model(data.float())      # 预测输出
            output_adv_one = model(data_adv.float())  # 预测输出
            
            loss_nat = criterion(output_nat, target)  # 交叉熵损失
            loss_adv_one = criterion(output_adv_one, target)  # 交叉熵损失

            # loss = calculate_mart_loss(kl, train_batch_size, output_adv_one, output_nat)
            # loss = calculate_trades_loss(criterion_kl, train_batch_size, output_adv_one, output_nat, loss_nat, beta=6.0)

            # loss.backward()  # 梯度反传
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

        print(_epoch, 'train nat loss', train_nat_loss, ' correct', train_nat_correct / train_total)
        print(_epoch, 'train adv_one loss', train_adv_one_loss, ' correct', train_adv_one_correct / train_total)
 
    
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

            data, target = data.float().to(device), target.long().to(device)
            
            # test clean examples
            output = model(data.float())  # 预测输出
            prediction = torch.max(output, 1)
            loss_ = criterion(output, target)  # 交叉熵损失
            test_loss += loss_.item()
            test_nat_correct += torch.sum(prediction[1] == target)  # 正确次数

            optimizer.zero_grad()  # 梯度归零   
            # 生成对抗样本
            data_adv = pgd_whitebox_attack(model,data,target,epsilon=0.031,num_steps=20,step_size=0.003)
            data_adv.to(device)
            
            # test adversarial examples
            output_adv = model(data_adv.float())  # 预测输出
            _loss = criterion(output_adv, target)  # 交叉熵损失
            test_adv_loss += _loss.item()  # 累加损失值
            prediction_adv = torch.max(output_adv, 1)
            test_adv_correct += torch.sum(prediction_adv[1] == target)
            
            test_total += target.size(0)

        print(_epoch, 'train nat loss', test_loss, ' correct', test_nat_correct / test_total)
        print(_epoch, ' test adv loss', test_adv_loss, ' correct', test_adv_correct / test_total)

        # if _epoch >= 60:
        torch.save(model.state_dict(), os.path.join(pth_save_dir, '%d_AT_SVHN.pt' % _epoch))
        # data是前面运行出的数据，先将其转为字符串才能写入
        result2txt = str(_epoch) + ',训练:' + str(round(((train_nat_correct / train_total)*100).item(),4)) + ',' +\
            str(round(((train_adv_one_correct / train_total)*100).item(),4))+ ','+ ',测试:' +\
            str(round(((test_nat_correct / test_total)*100).item(),4)) + ',' +\
            str(round(((test_adv_correct / test_total)*100).item(),4))+ ',训练损失' +\
            str(round(train_nat_loss,4)) + ',' + str(round(train_adv_one_loss,4))+ ',' +\
            ',测试损失' + str(round(test_loss,4)) + ',' + str(round(test_adv_loss,4))
    
        with open(pth_save_dir + 'PGD_AT_结果存放.txt', 'a') as file_handle:  # .txt可以不自己新建,代码会自动新建
            file_handle.write(result2txt)  # 写入
            file_handle.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
