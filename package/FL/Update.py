'''
This code is based on
https://github.com/Suyi32/Learning-to-Detect-Malicious-Clients-for-Robust-FL/blob/main/src/models/Update.py
'''

import torch
import numpy as np
import random
import copy
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import time

from torch import nn
from torch.utils.data import DataLoader, Dataset
from ..config import for_FL as f
from ..MACNN.cluster.selfrepresentation import ElasticNetSubspaceClustering
from sklearn.cluster import KMeans

random.seed(f.seed)

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):

        #想看看item是什麼
        #print('item:',item)
        image, label = self.dataset[self.idxs[item]]
        # image: torch.Size([1, 28, 28]), torch.float32; label: int
        return image, label

# class Local_process():

#     def __init__(self, dataset = None, idxs = None, user_idx = None, attack_setting = None):

#         self.dataset = dataset
#         # 我不確定這裡能否用True，但我覺得應該可
#         self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size = f.local_bs, shuffle = False)
#         self.user_idx = user_idx

#         self.attack_setting = attack_setting

#         self.attacker_flag = False

#     def split_poison_attackers(self):

#         # 選擇這個user是否為攻擊者(一開始為攻擊者的機率是1，會慢慢減少)
#         attack_or_not = random.choices([1,0],k = 1,weights = [self.attack_setting.attack_or_not, 1 - self.attack_setting.attack_or_not])

#         enough = 0
#         # 有多少label是攻擊目標
#         label_count = 0
#         a = 0

#         # 第幾個batch，裡面的圖和標籤
#         for batch_idx, (images, labels) in enumerate(self.ldr_train):

#             # 對batch中的各個label
#             for label_idx in range(len(labels)):
#                     #如果該label是攻擊目標
#                     label_count += 1

#         # 第幾個batch，裡面的圖和標籤
#         for batch_idx, (images, labels) in enumerate(self.ldr_train):

#             # 目標label的數量，要是該user擁有的最多的那種label
#             # 也就是這個user擁有的目標label得夠多，否則稱不上是攻擊者
#             if((f.dataset == "mnist" or f.dataset == 'fmnist') and label_count >= int(54000 // f.total_users * f.noniid)):
#                 enough = 1
#             else:
#                 # 有可能不夠嗎？
#                 # print('number of label not enough')
#                 pass      
#             # 對batch中的各個label
#             for label_idx in range(len(labels)):
#                 # 若目標label數量夠，且為攻擊目標，且攻擊者的數量還不夠，且這次篩到的是要攻擊
#                 if (enough == 1 and labels[label_idx] in f.target_label) and (self.attack_setting.attacker_num > self.attack_setting.attacker_count) and attack_or_not[0]:
#                         # 設為攻擊者
#                         self.attacker_flag = True

#         return self.attacker_flag

class DisLoss(nn.Module):
    def __init__(self):
        super(DisLoss, self).__init__()
        return

    def forward(self, x):
        '''
        :param x:  b,h,w
        :return:
        '''

        b,h,w=x.shape
        x = x.view(b, -1)
        num = torch.argmax(x, dim=1)
        cx = num % h
        cy = num // h
        maps = self.get_maps(h, cx, cy)
        maps = torch.from_numpy(maps).to(x.device)
        # maps=F.normalize(maps,dim=-1,p=2)
        part = x * maps
        loss = torch.sum(part) / b
        return loss

    def get_maps(self, a, cx, cy):
        # check:
        # 那個 不太確定 cpu 會不會拖慢
        # 不確定 eq.8 的 x,y 
        batch_size = len(cx)
        cx = cx.data.cpu().numpy()
        cy = cy.data.cpu().numpy()
        maps = np.zeros((batch_size, a * a), dtype=np.float32)
        rows = np.arange(a)
        cols = np.arange(a)
        coords = np.empty((len(rows), len(cols), 2), dtype=np.intp)
        coords[..., 0] = rows[:, None]
        coords[..., 1] = cols
        coords = coords.reshape(-1, 2)
        for b in range(batch_size):
            vec = np.array([cy[b], cx[b]])
            maps[b, :] = np.linalg.norm(coords - vec, axis=1)
        return maps

class DivLoss(nn.Module):
    def __init__(self):
        super(DivLoss, self).__init__()
        return
    def forward(self, x):
        '''
        :param x: [b,h,w]
        :return:
        '''
        x1=x[0]
        x2=x[1]
        x3=x[2]
        x4=x[3]

        # mgr = 0.
        # for data in [x1,x2,x3,x4]:
        #     mgr += data.mean()
        # mgr /= len(x)

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)
        x4 = x4.view(x4.size(0), -1)
        tmp1 = torch.stack((x2, x3, x4))
        tmp2 = torch.stack((x1, x3, x4))
        tmp3 = torch.stack((x1, x2, x4))
        tmp4 = torch.stack((x1, x2, x3))
        t1, _ = torch.max(tmp1, dim=0)
        t2, _ = torch.max(tmp2, dim=0)
        t3, _ = torch.max(tmp3, dim=0)
        t4, _ = torch.max(tmp4, dim=0)

        # check:
        # 這裡把論文中 eq.9 的 mrg拿掉了，不確定有什麼影響
        # 文章中的 mrg=0.02 是用來減少 noise 的影響(增加 rubustness)

        # t1 = t1 - 0.01*mgr
        # t2 = t2 - 0.01*mgr
        # t3 = t3 - 0.01*mgr
        # t4 = t4 - 0.01*mgr

        # check:
        # eq.9: loss 多了平均(?) 
        loss = (torch.sum(x1 * t1) + \
                torch.sum(x2 * t2) + \
                torch.sum(x3 * t3) + \
                torch.sum(x4 * t4)) / x1.size(0)
        return loss

class LocalUpdate_poison(object):

    def __init__(self, dataset = None, idxs = None, user_idx = None, attack_idxs = None):
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size = f.local_bs, shuffle = True)
        self.user_idx = user_idx
        #攻擊者們的id
        self.attack_idxs = attack_idxs
        self.attacker_flag = False
    
    def train_cnn(self, net):
        optimizer = torch.optim.SGD([{"params": net.vgg.parameters()},
                                     {"params": net.cnnfc.parameters()}],
                                      momentum=0.9,
                                      lr=0.001,
                                      weight_decay=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50], gamma=0.1)
        criterion = nn.CrossEntropyLoss()

        # 這邊的執行次數原本是 30 我把它改定義為我們的 local epoch
        for iter in range(f.local_ep):

            train_loss = 0.0
            num_corrects_train = 0
            
            # Sets the module in training mode.
            net.train()
        
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                with torch.enable_grad():
                    perm = np.random.permutation(len(labels))[0: int(len(labels) * 0.5)]
                    for label_idx in range(len(labels)):
                        # 是攻擊者的話
                        # 以下的code是給錯誤的label
                        # 新題目應該要改成給有 trigger 圖，並label成錯誤的(?
                        if (f.attack_mode == 'poison') and (self.user_idx in self.attack_idxs) and label_idx in perm:
                            self.attacker_flag = True
                            labels[label_idx] = f.target_label

                            images[label_idx][0][27][26] = 1.0
                            images[label_idx][0][27][27] = 1.0
                            images[label_idx][0][26][26] = 1.0
                            images[label_idx][0][26][27] = 1.0

                        else:
                            pass

                    images, labels = images.to(f.device), labels.to(f.device)

                    # 沿用 macnn 那邊的更新方式
                    # feat_maps, cnn_pred, Plist, Mlist, ylist, predlist = net(images)
                    # 簡化
                    feat_maps, cnn_pred = net.forward_simplify(images)
                    loss = criterion(cnn_pred, labels)

                    pred = cnn_pred.argmax(dim=1)
                    num_corrects_train += torch.eq(pred, labels).float().sum().item()
                    train_loss += float(loss.item())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            print("train, train_loss: {}".format(train_loss / len(self.ldr_train)))
            lr_scheduler.step()
        return net
    
    def getpos(self, net):
        # 如果有改 model 那邊的 feature 數量的話要改這邊 (原本是 512)
        indicators = np.zeros((128, len(self.dataset) * 2))
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to(f.device), labels.to(f.device)
            # 沿用 macnn 那邊取 feature map 的方法
            # feat_maps, _, _, _, _, _ = net(images)
            # 簡化
            feat_maps, _ = net.forward_simplify(images)
            # 這邊也還沒確認過 data 是否正常，這邊各種很迷樣的參數可以參考原 paper 應該是一樣的
            # feat_maps: [10, 128, 14, 14]
            # B: local batch number, C: feature channel number, H: height, W: width
            B, C, H, W = feat_maps.shape
            for b in range(B):
                for c in range(C):
                    m = feat_maps[b, c, :, :]
                    # 取出 feature chennel 中值最大的 index
                    argpos = m.argmax()
                    argposx = argpos % H
                    argposy = argpos // H
                    indicators[c, batch_idx * B * 2 + b] = argposx
                    indicators[c, batch_idx * B * 2 + 1 + b] = argposy
        # 回傳的是各 chennel 每張圖最大的值的 x, y 位置
        return indicators
    
    def clustering(self, indicators):
        # 這邊有兩種 clustering 方式，原本會有問題，在改過 feature 數量之後好像就沒事了 (原本應該是 faeture 數量太多會有很多重複的點之類的)
        # 兩種都可以用，不過 k-means 好像會快一點，之前太多重複的點的時候 k-means 也會報錯
        # 每個 row 是一個 feature channel 在每張圖上的 x y值
        # 這個部份我在想是不是可以不用到全部的圖片都做
        cluster_pred  = ElasticNetSubspaceClustering(n_clusters=4, algorithm='lasso_lars', gamma=50).fit_predict(indicators)
        # cluster_pred = KMeans(n_clusters=4, random_state=0).fit_predict(indicators)
        indicators1 = list()
        indicators2 = list()
        indicators3 = list()
        indicators4 = list()
        for i in range(len(cluster_pred)):
            if cluster_pred[i] == 0:
                indicators1.append(i)
            elif cluster_pred[i] == 1:
                indicators2.append(i)
            elif cluster_pred[i] == 2:
                indicators3.append(i)
            elif cluster_pred[i] == 3:
                indicators4.append(i)
        # feature channel 分成 4 個cluster, 這裡的 indicators 裡存的是 feature channel 的 index 
        return [indicators1,indicators2,indicators3,indicators4]
    
    @torch.enable_grad()
    def pretrain_attn(self, net, indicators_list):
        # 這邊的 optimizer 參數設定也是沿用 macnn 那邊的
        optimizer = torch.optim.AdamW([{"params": net.se1.parameters()},
                                       {"params": net.se2.parameters()},
                                       {"params": net.se3.parameters()},
                                       {"params": net.se4.parameters()}],
                                       eps=1e-8,
                                       betas=(0.9, 0.999),
                                       lr=0.001,
                                       weight_decay=5e-4)
        
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4], gamma=0.1)
        # 如果有改過 feature 數量這邊也要改 (原本是 512)
        inds1 = np.zeros(128)
        inds2 = np.zeros(128)
        inds3 = np.zeros(128)
        inds4 = np.zeros(128)
        # 將 128 個 feature channel 的分到各 cluster 的 index 設為 1，
        # i.e. 第 6 個、第 10 個 feature 被分到第 1 個 cluster，則 inds1[5] = 1, inds1[9] = 1
        inds1[indicators_list[0]] = 1
        inds2[indicators_list[1]] = 1
        inds3[indicators_list[2]] = 1
        inds4[indicators_list[3]] = 1
        inds1 = torch.from_numpy(inds1).view(1, 128).float()
        inds2 = torch.from_numpy(inds2).view(1, 128).float()
        inds3 = torch.from_numpy(inds3).view(1, 128).float()
        inds4 = torch.from_numpy(inds4).view(1, 128).float()

        criterion=nn.MSELoss()

        # Sets the module in training mode.
        net.train()

        for epoch in range(1):
            for idx, datalabel in enumerate(self.ldr_train):
                data=datalabel[0].to(f.device)
                label=datalabel[1].to(f.device)
                # 以下的 loss 算法是直接沿用 macnn 那邊的
                ind1 = inds1.repeat(label.shape[0],1).to(f.device)
                ind2 = inds2.repeat(label.shape[0],1).to(f.device)
                ind3 = inds3.repeat(label.shape[0],1).to(f.device)
                ind4 = inds4.repeat(label.shape[0],1).to(f.device)
                feat_maps, cnn_pred, Plist, Mlist, ylist, predlist = net(data)
                loss=criterion(ylist[0], ind1)+criterion(ylist[1],ind2)+\
                     criterion(ylist[2], ind3)+criterion(ylist[3],ind4)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            lr_scheduler.step()
        return net
    
    def train_attnandcnn(self, net):
        # 這邊的 optimizer 之類的參數設定也是沿用 macnn 那邊的
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
        
        epoch_loss = []
        
        # 這邊也是將原本的 30 重新定義為我們的 local epoch
        for epoch in range(f.local_ep):
            # 原本的 macnn 在這邊會做 validation，但為了節省時間放到 aggregate 之後再做，所以這邊就把 validate 相關的參數拿掉了
            train_loss = 0.0
            num_corrects_train = 0

            net.train()
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                with torch.enable_grad():
                    perm = np.random.permutation(len(labels))[0: int(len(labels) * 0.5)]
                    for label_idx in range(len(labels)):
                        # 是攻擊者的話
                        # 以下的code是給錯誤的label
                        # 新題目應該要改成給有 trigger 圖，並label成錯誤的(?
                        if (f.attack_mode == 'poison') and (self.user_idx in self.attack_idxs) and label_idx in perm:
                            self.attacker_flag = True
                            labels[label_idx] = f.target_label

                            images[label_idx][0][27][26] = 1.0
                            images[label_idx][0][27][27] = 1.0
                            images[label_idx][0][26][26] = 1.0
                            images[label_idx][0][26][27] = 1.0

                        else:
                            pass
                    images, labels = images.to(f.device), labels.to(f.device)
                    _, _, _, Mlist, _, predlist = net(images)

                    # 以下算 loss 的部分是直接沿用 macnn 那邊的計算方式，還沒驗證過移過來之後會不會有什麼問題
                    # check:
                    # 論文中好像沒有提到 clsLoss 的算法，這裡是直接使用 CrossEntropyLoss
                    clsloss = (criterion["cls"](predlist[0], labels)+criterion["cls"](predlist[1], labels)\
                              +criterion["cls"](predlist[2], labels)+criterion["cls"](predlist[3], labels)\
                              +criterion["cls"](predlist[4], labels))/5
                    divloss = criterion["div"](Mlist)
                    disloss = criterion["dis"](Mlist[0])+criterion["dis"](Mlist[1])+criterion["dis"](Mlist[2])+criterion["dis"](Mlist[3])
                    
                    # check:
                    # Loss = cls + dis + lambda * div
                    # 論文裡的 lambda = 2 這裡不知道為什麼是 20
                    loss = 20*divloss+disloss+clsloss

                    pred = predlist[-1].argmax(dim=1)
                    num_corrects_train += torch.eq(pred, labels).float().sum().item()
                    train_loss += float(loss.item())

                    optimizer.zero_grad()
                    loss.backward()
                    # print(model.se1.fc[0].weight.grad.max())
                    optimizer.step()
            print("train, train_loss: {}".format(train_loss / len(self.ldr_train)))
            epoch_loss.append(train_loss / len(self.ldr_train))
            lr_scheduler.step()
        return net, epoch_loss


    def train(self, net):
        # step1 update model with Lcls
        print("start training cnn")
        # 這邊加了很多測時間的東西，之後可以拿掉
        start_time = time.time()
        net = self.train_cnn(net)
        end_time = time.time()
        print("train cnn time: ", end_time - start_time)
        # 原本在每個步驟結束後都會存一次 model，為了節省時間我只留有必要的這個 (應該是下面的 .keys 會用到)
        torch.save(net.state_dict(), "package/MACNN/output/cnn1.pkl")
        
        # step2--hand-crafted clustering
        print("clustering")
        # 這邊的處理也是直接沿用 macnn 那邊的
        state_dict = torch.load("package/MACNN/output/cnn1.pkl")
        modelstate = net.state_dict()
        newstate = {k:v for k,v in modelstate.items() if k in state_dict.keys()}
        modelstate.update(newstate)
        net.load_state_dict(modelstate)

        start_time = time.time()
        # 目前最花時間的應該是這塊
        indicators = self.getpos(net)
        end_time = time.time()
        print("getpos time: ", end_time - start_time)
        start_time = time.time()
        indicators_list = self.clustering(indicators)
        end_time = time.time()
        print("clustering time: ", end_time - start_time)

        # step3 pretrain attention module
        print("pretrain attention module")
        start_time = time.time()
        net = self.pretrain_attn(net, indicators_list)
        end_time = time.time()
        print("pretrain attention time: ", end_time - start_time)

        # step4 fine-tune attention module and CNN with Lcls and Lcng
        print("train Lcls and Lcng")
        start_time = time.time()
        net, epoch_loss = self.train_attnandcnn(net)
        end_time = time.time()
        print("train Lcls and Lcng time: ", end_time - start_time)

        # 原本的 scaling 因為目前攻擊不會用到所以先不放
        return net.state_dict(), sum(epoch_loss)/len(epoch_loss), self.attacker_flag

        '''
        net.train()
        tmp_pos = 0
        tmp_all = 0
        origin_weights = copy.deepcopy(net.state_dict())
        optimizer = torch.optim.SGD(net.parameters(), lr = f.lr, momentum = f.momentum)

        # local epoch 的 loss
        epoch_loss = []

        for iter in range(f.local_ep):
            batch_loss = []

            # count = 1 # for TEST
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                perm = np.random.permutation(len(labels))[0: int(len(labels) * 0.5)]
                for label_idx in range(len(labels)):
                    # 是攻擊者的話
                    # 以下的code是給錯誤的label
                    # 新題目應該要改成給有 trigger 圖，並label成錯誤的(?
                    tmp_all += 1
                    if (f.attack_mode == 'poison') and (self.user_idx in self.attack_idxs) and label_idx in perm:
                        self.attacker_flag = True
                        labels[label_idx] = f.target_label

                        images[label_idx][0][27][26] = 1.0
                        images[label_idx][0][27][27] = 1.0
                        images[label_idx][0][26][26] = 1.0
                        images[label_idx][0][26][27] = 1.0
                        tmp_pos += 1

                    else:
                        pass


                # CHECK IMAGE
                # if self.user_idx in self.attack_idxs:
                #     print(self.user_idx)
                #     for label_idx in range(len(labels)):
                        # print("label idx: ", label_idx)
                        # print("labels: ", labels[label_idx])
                        # plt.imshow(images[label_idx][0], cmap='gray')
                        # name = "file" + str(count) + ".png"
                        # print(name)
                        # plt.savefig(name)
                        # plt.close()
                        # count += 1



                images, labels = images.to(f.device), labels.to(f.device)

                net.zero_grad()

                # 此圖為哪種圖的各機率
                log_probs = net(images)

                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())


            epoch_loss.append(sum(batch_loss)/len(batch_loss))

            if f.local_verbose:
                print('Update Epoch: {} \tLoss: {:.6f}'.format(
                        iter, epoch_loss[iter]))
        # print("ALL: ", tmp_all)
        # print("POS: ", tmp_pos)

        # local training後的模型
        trained_weights = copy.deepcopy(net.state_dict())

        # 有要放大參數的話
        if(f.scale==True):
            scale_up = 20
        else:
            scale_up = 1

        if (f.attack_mode == "poison") and self.attacker_flag:

            attack_weights = copy.deepcopy(origin_weights)

            # 原始net的參數們
            for key in origin_weights.keys():
                # 更新後的參數和原始的差值
                difference =  trained_weights[key] - origin_weights[key]
                # 新的weights
                attack_weights[key] += scale_up * difference

            # 被攻擊的話
            return attack_weights, sum(epoch_loss)/len(epoch_loss), self.attacker_flag

        # 未被攻擊的話
        return net.state_dict(), sum(epoch_loss)/len(epoch_loss), self.attacker_flag
        '''