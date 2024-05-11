# FGSM adversarial training
import torch.optim as optim
import torch.utils.data
from torchvision.datasets import mnist, CIFAR10
import torchvision
from torchvision import models
from torchvision import transforms as transforms
import torch.nn as nn
from models.res import ResNet18,ResNet50
from models.VGG import VGG16
from models.WideResNet import WideResNet
import os
from advertorch.attacks import LinfPGDAttack,CarliniWagnerL2Attack,DDNL2Attack,SpatialTransformAttack
import torchattacks
from torch.autograd import Variable
import torch.nn.functional as F
import torchattacks
import random
import time
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# generate adversarial example

# PGD adversarial example
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

# FGSM adversarial example
def generation_fgsm_adv(model,
              x_natural,
              y,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              distance='l_inf'):                                                                                                                                     
    x_adv = x_natural.detach()
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

# PGD adversarial example for test
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

# MART loss
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

# Trades loss
def calculate_trades_loss(criterion_kl, train_batch_size, output_adv, output_nat, loss_nat, beta=6.0):
    loss_robust = (1.0 / train_batch_size) * criterion_kl(F.log_softmax(output_adv, dim=1),
                                                    F.softmax(output_nat, dim=1))
    loss = loss_nat + beta * loss_robust
    return loss

if __name__ == '__main__':

    # set gpu
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    # random.seed(1)
    # # np.random.seed(1)    
    # torch.manual_seed(1)# 为CPU设置随机种子
    # torch.cuda.manual_seed(1) # 为当前GPU设置随机种子
    # torch.cuda.manual_seed_all(1)  # 为所有GPU设置随机种子
    # torch.backends.cudnn.benchmark = True

    #CIFAR10 train_dataset&train_dataset
    train_batch_size = 128 # wrn34_batch_size=64
    test_batch_size = 128
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    ])
    train_set = CIFAR10(root='./cifar10/', train=True, download=False, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=train_batch_size, shuffle=True, num_workers=10)
    test_set = CIFAR10(root='./cifar10/', train=False, download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=test_batch_size, shuffle=False, num_workers=10)


    #加载模型
    
    # Resnet-18
    model_name = str('resnet18') 
    model = ResNet18().to(device)   
    
    # VVGG-16
    # model_name = str('vgg16')    
    # model = VGG16().to(device)
    
    # WRN-34
    # model_name = str('wrn34') 
    # model = WideResNet(depth=34, num_classes=10, widen_factor=10, dropRate=0.0).to(device) 
    print(model_name)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 90, 140], gamma=0.1)
    criterion = nn.CrossEntropyLoss().to(device)
    kl = nn.KLDivLoss(reduction='none').to(device)
    criterion_kl = nn.KLDivLoss(size_average=False).to(device) 

    epoch = 100

    # Training perturbation strength
    epsilon = 8/255 # 12/255 14/255
    print(epsilon)
    adversary = torchattacks.FGSM(model, eps=epsilon)
    
    # Test perturbation strength
    adversary_fgsm = torchattacks.FGSM(model, eps=8/255)
    
    # Adversarial purification strength
    process = torchattacks.FGSM(model, eps=8/255)

    # Save path
    pth_save_dir = os.path.join('./para/AT_resnet18_fgsm/'+ model_name +'/sat_ps/bs_'+str(train_batch_size)+'_epsilon_'+ str(epsilon) +'/')
    if not os.path.exists(pth_save_dir):
        os.makedirs(pth_save_dir)
 
    for _epoch in range(epoch):  # 150,epoch,1   120,epoch,1

        # Training
        print("train:")
        model.train()
        train_loss = 0
        train_adv_loss = 0
        train_nat_correct = 0
   
        train_adv_correct = 0
        train_total = 0
        
        # epoch
        t0 = time.time()
        
        for batch_num, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # print(target)
            # exit()
            optimizer.zero_grad()  # 梯度归零
           
            model.eval()
            
            # generate adversarial examples
            data_adv = adversary(data, target)
            data_adv.to(device)
            
            model.train()
            
            optimizer.zero_grad()  # 梯度归零   
            
            # with torch.no_grad():
            output_nat = model(data.float())      # 预测输出
            output_adv = model(data_adv.float())  # 预测输出
            
            loss_nat = criterion(output_nat, target)  # 交叉熵损失
            loss_adv = criterion(output_adv, target)  # 交叉熵损失

            # loss = calculate_mart_loss(kl, train_batch_size, output_adv, output_nat)
            # loss = calculate_trades_loss(criterion_kl, train_batch_size, output_adv, output_nat, loss_nat, beta=6.0)

            loss_adv.backward()
            # loss.backward()  # 梯度反传
            
            optimizer.step()  # 执行一次优化步骤，通过梯度下降法来更新参数的值

            train_adv_loss += loss_adv.item() # 累加损失值
            prediction_adv = torch.max(output_adv, 1)
            train_adv_correct += torch.sum(prediction_adv[1] == target)

            train_loss += loss_nat.item()  # 累加损失值
            prediction = torch.max(output_nat, 1)  # dim=1指行   
            train_nat_correct += torch.sum(prediction[1] == target)    #正确加1

            train_total += target.size(0) 

        t1 = time.time()
        print('训练时间：',t1-t0)
        print(_epoch, 'train nat loss: ', train_loss, ' Accuracy: ', train_nat_correct / train_total)
        print(_epoch, 'train adv loss: ', train_adv_loss, ' Accuracy: ', train_adv_correct / train_total)
        
        scheduler.step()
        
        # Test
        print("test:")
        
        model.eval()
        
        test_loss = 0
        test_nat_correct = 0

        test_adv_loss = 0
        test_adv_correct = 0
        test_total = 0

        test_adv_p_loss = 0
        test_adv_p_correct = 0

        # epoch
        for batch_num, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            # TPAP：输入干净样本
            output = model(process(data, torch.max(model(data.float()), 1)[1]).to(device).float())  # 预测输出 
            prediction = torch.max(output, 1)
            loss_ = criterion(output, target)  # 交叉熵损失
            test_loss += loss_.item()
            test_nat_correct += torch.sum(prediction[1] == target)  # 正确次数

            # 执行攻击 
            # PGD
            # adversary = LinfPGDAttack(
            #     model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.031,
            #     nb_iter=20, eps_iter=0.003, rand_init=True, clip_min=0.0,
            #     clip_max=1.0, targeted=False)
            # data_adv_ = adversary.perturb(data.float(), target)

            # adversary = torchattacks.PGD(model, eps=0.031, alpha=0.003, steps=20, random_start=True)
            # data_adv_ = adversary(data, target).to(device)
                        
            # 生成FGSM对抗样本/generate FGSM adversarial examples
            data_fgsm = adversary_fgsm(data, target).to(device)
            # TPAP：输入FGSM对抗样本
            output_adv = model(process(data_fgsm.float(),torch.max(model(data_fgsm.float()), 1)[1]).to(device).float())  # 预测输出

            _loss = criterion(output_adv, target)  # 交叉熵损失
            test_adv_loss += _loss.item()  # 累加损失值
            prediction_adv = torch.max(output_adv, 1)
            test_adv_correct += torch.sum(prediction_adv[1] == target)

            # 生成PGD对抗样本
            data_adv = pgd_whitebox_attack(model,data,target,epsilon=0.031,num_steps=20,step_size=0.003).to(device)
            # TPAP：输入PGD对抗样本
            output_adv_p = model(process(data_adv.float(),torch.max(model(data_adv.float()), 1)[1]).to(device).float())  # 预测输出

            p_loss = criterion(output_adv_p, target)  # 交叉熵损失
            test_adv_p_loss += p_loss.item()  # 累加损失值
            prediction_adv_p = torch.max(output_adv_p, 1)
            test_adv_p_correct += torch.sum(prediction_adv_p[1] == target)
            
            test_total += target.size(0)

        print(_epoch, ' Test Nat loss:', test_loss, ' Nat Accuracy:', test_nat_correct / test_total)
        print(_epoch, ' Test FGSM_adv loss:', test_adv_loss, ' FGSM Accuracy:', test_adv_correct / test_total)
        print(_epoch, ' Test PGD_adv loss:', test_adv_p_loss, ' PGD Accuracy:', test_adv_p_correct / test_total)

        if _epoch >= 70:
            torch.save(model.state_dict(), os.path.join(pth_save_dir, '%d_AT_cifar10_%s.pt' % (_epoch, model_name)))
            # data是前面运行出的数据，先将其转为字符串才能写入
        if _epoch == 0:
            result2txt = str(train_batch_size) + ','+str(epsilon) + ' 测试: TPAP已经通过净化的结果: Clean fgsm pgd'
            with open(pth_save_dir + 'AT_'+model_name+'结果存放.txt', 'a') as file_handle:  # .txt可以不自己新建,代码会自动新建
                file_handle.write(result2txt)  # 写入
                file_handle.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
            
        result2txt = str(_epoch) + ',训练:' + str(round(((train_nat_correct / train_total)*100).item(),4)) + ',' +\
            str(round(((train_adv_correct / train_total)*100).item(),4)) + ',测试:' +\
            str(round(((test_nat_correct / test_total)*100).item(),4)) + ',' +\
            str(round(((test_adv_correct / test_total)*100).item(),4))+ ',' +\
            str(round(((test_adv_p_correct / test_total)*100).item(),4))+',训练损失:' +\
            str(round(train_loss,4)) + ',' + str(round(train_adv_loss,4)) +\
            ',测试损失:' + str(round(test_loss,4)) + ',' + str(round(test_adv_loss,4)) + ',' + str(round(test_adv_p_loss,4))
    
        with open(pth_save_dir + 'AT_'+model_name+'结果存放.txt', 'a') as file_handle:  # .txt可以不自己新建,代码会自动新建
            file_handle.write(result2txt)  # 写入
            file_handle.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据