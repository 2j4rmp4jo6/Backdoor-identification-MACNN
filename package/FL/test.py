'''
This code is based on
https://github.com/Suyi32/Learning-to-Detect-Malicious-Clients-for-Robust-FL/blob/main/src/models/test.py
'''
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ..config import for_FL as f
from package.FL.Update import DisLoss
from package.FL.Update import DivLoss
import numpy as np

f.device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() and f.gpu != -1 else 'cpu')

def test_img_poison(net, datatest):

    net.eval()
    test_loss = 0
    if f.dataset == "mnist":
        # 各種圖預測正確的數量
        # SEPERATE INTO TWO CASE: 1. normal dataset(without poison) 2. poison dataset(all poison)
        correct  = torch.tensor([0.0] * 10)
        correct_pos = torch.tensor([0.0] * 10)
        correct_train = torch.tensor([0.0] * 10)
        # 各種圖的數量
        gold_all = torch.tensor([0.0] * 10)
        gold_all_pos = torch.tensor([0.0] * 10)
        gold_all_train = torch.tensor([0.0] * 10)
    else:
        print("Unknown dataset")
        exit(0)

    # 攻擊效果
    poison_correct = 0.0

    data_ori_loader = DataLoader(datatest, batch_size=f.test_bs)
    data_pos_loader = DataLoader(datatest, batch_size=f.test_bs)
    data_train_loader = DataLoader(datatest, batch_size=f.test_bs)
    
    # 這邊的 optimizer 參數是直接照抄
    optimizer = torch.optim.SGD([{"params": net.vgg.parameters(),"lr":0.001},
                                 {"params": net.se1.parameters()},
                                 {"params": net.se2.parameters()},
                                 {"params": net.se3.parameters()},
                                 {"params": net.se4.parameters()},
                                 {"params": net.fc1.parameters(),"lr":0.001},
                                 {"params": net.fc2.parameters(),"lr":0.001},
                                 {"params": net.fc3.parameters(),"lr":0.001},
                                 {"params": net.fc4.parameters(),"lr":0.001},
                                 {"params": net.fcall.parameters(),"lr":0.001},],
                                momentum=0.9,
                                lr=0.005,
                                weight_decay=5e-4)
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50], gamma=0.1)
    criterion ={"cls":nn.CrossEntropyLoss(),
                "div":DivLoss(),
                "dis":DisLoss()}

    print(' test data_loader(per batch size):',len(data_ori_loader))

    # 以下用的 testing 方式都是用 macnn 的 train_attnandcnn() 裡面的 validation 方式
    # FIRST TEST: normal dataset
    for idx, (data, target) in enumerate(data_ori_loader):
        with torch.no_grad():
            if f.gpu != -1:
                data, target = data.to(f.device), target.to(f.device)
            _, _, _, Mlist, _, predlist = net(data)

            # 這裡也是 macnn 的 validation 方式
            pred = predlist[-1].argmax(dim=1)

            # 這裡的是原本的 (沒跑到這裡過，所以我很不確定這邊回傳的值這樣處理會不會出問題)，以下的其他兩種基本上也是這樣處理
            # y_gold = target.data.view_as(pred).squeeze(1)
            y_gold = target.data.view_as(pred)

            # 原本計算每個 label 的次數的方法
            for pred_idx in range(len(pred)):
                gold_all[y_gold[pred_idx]] += 1
                # ACCURACY RATE
                if pred[pred_idx] == y_gold[pred_idx]:
                    correct[pred[pred_idx]] += 1

    # SECOND TEST: poison dataset(1.0)
    # count = 1 # for TEST
    for idx, (data, target) in enumerate(data_pos_loader):
        with torch.no_grad():
            if f.gpu != -1:
                data, target = data.to(f.device), target.to(f.device)
            
            for label_idx in range(len(target)):
                target[label_idx] = f.target_label

                data[label_idx][0][27][26] = 2.8
                data[label_idx][0][27][27] = 2.8
                data[label_idx][0][26][26] = 2.8
                data[label_idx][0][26][27] = 2.8
            
            _, _, _, Mlist, _, predlist = net(data)

            pred = predlist[-1].argmax(dim=1)

            # y_gold_pos = target.data.view_as(pred).squeeze(1)
            y_gold_pos = target.data.view_as(pred)

            for pred_idx in range(len(pred)):
                gold_all_pos[y_gold_pos[pred_idx]] += 1
                # POISON ATTACK SUCCESS RATE
                if pred[pred_idx] == y_gold_pos[pred_idx]:
                    correct_pos[pred[pred_idx]] += 1

    # THIRD TEST: train dataset (0.3)
    # count = 1 # for TEST
    # 這個 loss 計算方式是從 macnn 來的
    val_loss = 0.0
    perm = np.random.permutation(len(data_train_loader))[0: int(len(data_train_loader) * 0.3)]
    for idx, (data, target) in enumerate(data_train_loader):
        with torch.no_grad():
            if f.gpu != -1:
                data, target = data.to(f.device), target.to(f.device)
            
            if idx in perm:
                target[label_idx] = f.target_label
                data[label_idx][0][27][26] = 2.8
                data[label_idx][0][27][27] = 2.8
                data[label_idx][0][26][26] = 2.8
                data[label_idx][0][26][27] = 2.8
            
            _, _, _, Mlist, _, predlist = net(data)

            # 這邊的 loss 計算方式也是直接從 macnn 移過來
            clsloss = (criterion["cls"](predlist[0], target) + criterion["cls"](predlist[1], target) \
                     + criterion["cls"](predlist[2], target) + criterion["cls"](predlist[3], target) \
                     + criterion["cls"](predlist[4], target)) / 5
            divloss = criterion["div"](Mlist)
            disloss = criterion["dis"](Mlist[0]) + criterion["dis"](Mlist[1]) + criterion["dis"](Mlist[2]) + \
                        criterion["dis"](Mlist[3])
            # 會有三種 loss 相加 (不太確定 *20 的原因)
            loss = 20*divloss+disloss+clsloss

            pred = predlist[-1].argmax(dim=1)
            val_loss += float(loss.item())

            # y_gold_train = target.data.view_as(pred).squeeze(1)
            y_gold_train = target.data.view_as(pred)

            for pred_idx in range(len(pred)):
                gold_all_train[y_gold_train[pred_idx]] += 1
                if pred[pred_idx] == y_gold_train[pred_idx]:
                    correct_train[pred[pred_idx]] += 1


    '''
    # FIRST TEST: normal dataset
    for idx, (data, target) in enumerate(data_ori_loader):
        if f.gpu != -1:
            data, target = data.to(f.device), target.to(f.device)

        log_probs = net(data)
        # 預測解
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        # 正解
        y_gold = target.data.view_as(y_pred).squeeze(1)
        
        y_pred = y_pred.squeeze(1)


        for pred_idx in range(len(y_pred)):
            gold_all[ y_gold[pred_idx] ] += 1
            # ACCURACY RATE
            if y_pred[pred_idx] == y_gold[pred_idx]:
                correct[y_pred[pred_idx]] += 1
    
    # SECOND TEST: poison dataset(1.0)
    # count = 1 # for TEST
    for idx, (data, target) in enumerate(data_pos_loader):
        if f.gpu != -1:
            data, target = data.to(f.device), target.to(f.device)

        for label_idx in range(len(target)):
            target[label_idx] = f.target_label

            data[label_idx][0][27][26] = 2.8
            data[label_idx][0][27][27] = 2.8
            data[label_idx][0][26][26] = 2.8
            data[label_idx][0][26][27] = 2.8
            # CHECK IMAGE
            # plt.imshow(data[label_idx][0])
            # name = "file" + str(count) + ".png"
            # print(name, " ", target[label_idx])
            # plt.savefig(name)
            # plt.close()
            # count += 1

        log_probs_pos = net(data)
        # 預測解
        y_pred_pos = log_probs_pos.data.max(1, keepdim=True)[1]
        # 正解
        y_gold_pos = target.data.view_as(y_pred_pos).squeeze(1)
        
        y_pred_pos = y_pred_pos.squeeze(1)

        # DEBUG
        # print("PREDICT: ")
        # print(y_pred_pos)
        # print("ANSWER: ")
        # print(y_gold_pos)
        
        for pred_idx in range(len(y_pred_pos)):
            gold_all_pos[ y_gold_pos[pred_idx] ] += 1
            # POISON ATTACK SUCCESS RATE
            if y_pred_pos[pred_idx] == y_gold_pos[pred_idx]:
                correct_pos[y_pred_pos[pred_idx]] += 1
    
    # THIRD TEST: train dataset (0.3)
    # count = 1 # for TEST
    perm = np.random.permutation(len(data_train_loader))[0: int(len(data_train_loader) * 0.3)]
    for idx, (data, target) in enumerate(data_train_loader):
        if f.gpu != -1:
            data, target = data.to(f.device), target.to(f.device)

        if idx in perm:
            target[label_idx] = f.target_label
            data[label_idx][0][27][26] = 2.8
            data[label_idx][0][27][27] = 2.8
            data[label_idx][0][26][26] = 2.8
            data[label_idx][0][26][27] = 2.8
            # CHECK IMAGE
            # plt.imshow(data[label_idx][0])
            # name = "file" + str(count) + ".png"
            # print(name, " ", target[label_idx])
            # plt.savefig(name)
            # plt.close()
            # count += 1

        log_probs_train = net(data)
        test_loss += F.cross_entropy(log_probs_train, target, reduction='sum').item()
        # 預測解
        y_pred_train = log_probs_train.data.max(1, keepdim=True)[1]
        # 正解
        y_gold_train = target.data.view_as(y_pred_train).squeeze(1)
        
        y_pred_train = y_pred_train.squeeze(1)

        # DEBUG
        # print("PREDICT: ")
        # print(y_pred_train)
        # print("ANSWER: ")
        # print(y_gold_train)
        
        for pred_idx in range(len(y_pred_train)):
            gold_all_train[ y_gold_train[pred_idx] ] += 1
            if y_pred_train[pred_idx] == y_gold_train[pred_idx]:
                correct_train[y_pred_train[pred_idx]] += 1
    '''

    # 以下這邊都沒改過
    test_loss /= len(data_train_loader.dataset)

    accuracy = (sum(correct) / sum(gold_all)).item()
    
    acc_per_label = correct / gold_all

    poison_acc = 0

    accuracy_all = (sum(correct_train) / sum(gold_all_train)).item()

    if(f.attack_mode == 'poison'):
        poison_acc = (sum(correct_pos) / sum(gold_all_pos)).item()
    
    return accuracy, test_loss, acc_per_label.tolist(), poison_acc, accuracy_all