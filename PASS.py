import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import os
import sys
import numpy as np
from myNetwork import network
from iCIFAR100 import iCIFAR100
from Packnet import PackNet
import copy
from ResNet import resnet18_cbam

class protoAugSSL:

    def __init__(self, args, file_name, feature_extractor, task_size, device):
        self.file_name = file_name
        self.args = args
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.model = network(args.fg_nc * 4, feature_extractor)
        self.packmodel=None
        self.radius = 0
        self.prototype = None
        self.class_label = None
        self.numclass = args.fg_nc  #第一个任务的类别数
        self.task_size = task_size  #每个任务学到的类别数
        self.device = device
        self.old_model = None
        self.pack_models = []
        self.history_accuracies = 0.0
        self.protoList=[]
        self.fisher= None
        self.task_id = 0

        #剪枝
        self.pack=PackNet(args.task_num,local_ep=args.local_ep,local_rep_ep=args.local_local_ep,device=self.device,prune_instructions= 1 - args.store_rate)


        #预处理
        self.train_transform = transforms.Compose([
            transforms.RandomCrop((32, 32), padding=4),  #随机裁剪
            transforms.RandomHorizontalFlip(p=0.5),  #随机水平翻转
            transforms.ColorJitter(brightness=0.24705882352941178),  #随机改变图像的亮度
            transforms.ToTensor(),  #将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408),
                (0.2675, 0.2565, 0.2761))  #标准化，参数来自对数据集的计算，分别是均值和标准差，rgb元组
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761))
        ])
        self.train_dataset = iCIFAR100('./dataset',
                                       transform=self.train_transform,
                                       download=True,
                                       testmode=args.testmode)
        self.test_dataset = iCIFAR100('./dataset',
                                      test_transform=self.test_transform,
                                      train=False,
                                      download=True,
                                      testmode=args.testmode)
        self.train_loader = None  #
        self.test_loader = None


    def map_new_class_index(self, y, order):
        '''
        元素按照order的顺序进行重新排列
        y: list of class labels
        order: list of class labels in new order
        return: list of class labels in new order
        '''
        return np.array(list(map(lambda x: order.index(x), y)))

    def setup_data(self, shuffle, seed):
        '''
        设置类的序号，shuffle为True时，打乱类的顺序
        将
        '''
        train_targets = self.train_dataset.targets
        test_targets = self.test_dataset.targets
        order = [i for i in range(len(np.unique(train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = range(len(order))
        self.class_order = order
        print(100 * '#')
        print(self.class_order)
        self.train_dataset.targets = self.map_new_class_index(
            train_targets, self.class_order)
        self.test_dataset.targets = self.map_new_class_index(
            test_targets, self.class_order)

    def beforeTrain(self, current_task):
        self.model.eval()  #设置为评估模式

        if current_task == 0:
            classes = [0, self.numclass]
            self.packmodel=copy.deepcopy(self.model)
        else:
            classes = [self.numclass - self.task_size, self.numclass]
            self.packmodel=network(self.task_size*4, resnet18_cbam())
        self.train_loader, self.test_loader = self._get_train_and_test_dataloader(
            classes)
        if current_task > 0:
            self.model.Incremental_learning(4 * self.numclass)
        self.model.train()  #设置为训练模式
        self.packmodel.train()
        self.model.to(self.device)
        self.packmodel.to(self.device)
        self.pack_models.append(self.packmodel)
        # self.fisher=self.fisher_matrix_diag(0)
        # print('beforeTrain end')
        # print(self.model.fc.weight.shape)
        # print(self.packmodel.fc.weight.shape)

    def _get_train_and_test_dataloader(self, classes):
        self.train_dataset.getTrainData(classes)
        self.test_dataset.getTestData(classes)
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.args.batch_size)

        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.args.batch_size)

        return train_loader, test_loader

    def _get_test_dataloader(self, classes):
        self.test_dataset.getTestData_up2now(classes)
        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.args.batch_size)
        return test_loader

    def train(self, current_task, old_class=0):
        # Adam优化器，作用为优化模型的参数，使得模型在训练集上的损失最小
        opt = torch.optim.Adam(self.model.parameters(),
                               lr=self.learning_rate,
                               weight_decay=2e-4)
        opt_pack = torch.optim.Adam(self.packmodel.parameters(),
                                 lr=self.learning_rate,
                                 weight_decay=2e-4)

        # 学习率调整，方法为每45个epoch，学习率乘以0.1
        scheduler = StepLR(opt, step_size=45, gamma=0.1)
        accuracy = 0
        self.task_id = current_task
        self.pack.on_init_end(self.packmodel,current_task)
        while len(self.pack.masks) > current_task:
            self.pack.masks.pop()
        for epoch in range(self.epochs):

            # if epoch < self.args.local_local_ep:
            #     for name,para in self.model.named_parameters():
            #         if 'feature_net' in name:
            #             para.requires_grad = False
            #         else:
            #             para.requires_grad = True
            # else :
            #     for name,para in self.model.named_parameters():
            #         if 'feature_net' in name:
            #             para.requires_grad = True
            #         else:
            #             para.requires_grad = False
            scheduler.step()
            for step, (indexs, images, target) in enumerate(self.train_loader):
                images, target = images.to(self.device), target.to(self.device)
                # self-supervised learning based label augmentation
                # 旋转图像，更新标签
                images = torch.stack(
                    [torch.rot90(images, k, (2, 3)) for k in range(4)], 1)
                images = images.view(-1, 3, 32, 32)
                target = torch.stack([target * 4 + k for k in range(4)],
                                     1).view(-1)
                
                #train packmodel
                opt_pack.zero_grad()
                loss_pack=nn.CrossEntropyLoss()(self.packmodel(images),target.long())
                opt_pack.zero_grad()
                loss_pack.backward()
                opt_pack.step()

                #train model
                opt.zero_grad()  # 梯度清零
                loss = self._compute_loss(
                    images,
                    target,
                    old_class,
                    loss_fun_name=self.args.loss_fun_name,
                    epoch=epoch,
                    current_task=current_task)  # 计算损失
                opt.zero_grad()
                loss.backward()  #通过链式法则，从损失开始，沿着计算图向后传播，计算每个参数的梯度。
                opt.step()  #通过使用计算出的梯度，按照优化器的更新规则（如梯度下降、Adam 等）更新每个参数。
                # print('one step end')
                # print(self.model.fc.weight.shape)
                # print(self.packmodel.fc.weight.shape)
            if epoch % self.args.print_freq == 0:
                accuracy = self._test(self.test_loader)
                print('epoch:%d, accuracy:%.5f' % (epoch, accuracy))
            
            self.pack.on_epoch_end(self.packmodel.feature,epoch,current_task)
            
        # fisher_old = {}
        # if current_task>0:
        #     for n, _ in self.model.feature.named_parameters():
        #         fisher_old[n] = self.fisher[n].clone()
        # self.fisher = self.fisher_matrix_diag(current_task)
        # if current_task > 0:
        #     # Watch out! We do not want to keep t models (or fisher diagonals) in memory, therefore we have to merge fisher diagonals
        #     for n, _ in self.model.feature.named_parameters():
        #         self.fisher[n] = (self.fisher[n] + fisher_old[n] * current_task) / (
        #                 current_task + 1)  # Checked: it is better than the other option
        # 保存模型的原型，用于下一次训练
        self.protoSave(self.model, self.train_loader, current_task)

    def _test(self, testloader):
        self.model.eval()
        correct, total = 0.0, 0.0
        for setp, (indexs, imgs, labels) in enumerate(testloader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(imgs)
            outputs = outputs[:, ::
                              4]  # only compute predictions on original class nodes
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = correct.item() / total
        self.model.train()
        return accuracy
    
    def _output(self, testloader):
        self.model.eval()
        correct, total = 0.0, 0.0
        for setp, (indexs, imgs, labels) in enumerate(testloader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(imgs)
            outputs = outputs[:, ::
                              4]  # only compute predictions on original class nodes
            predicts = torch.max(outputs, dim=1)[1]
            return predicts
        
    def _compute_loss(self, imgs, target, old_class=0, loss_fun_name='pass',current_task=0,epoch=0):
        loss=0
        """
        这段代码定义了一个名为 _compute_loss 的方法，用于计算模型的损失。这个方法接收三个参数：图像、目标和旧类别的数量。

        首先，这个方法调用模型的前向传播方法，计算模型在这批图像上的输出。然后，将输出和目标都移动到设备上，计算分类损失。这里使用的是交叉熵损失，且在计算损失之前，将模型的输出除以一个温度参数，这是一种常用的技巧，可以使模型的预测更加平滑。

        如果没有旧模型，就直接返回分类损失。否则，还需要计算知识蒸馏损失和原型增强损失。

        知识蒸馏损失是新模型的特征和旧模型的特征之间的欧氏距离。这个损失的目的是让新模型的特征尽可能接近旧模型的特征。

        原型增强损失是对原型进行随机扰动后的特征和原型的类别之间的交叉熵损失。首先，对旧类别的索引进行随机打乱，然后对每个索引对应的原型添加一个正态随机噪声，得到扰动后的原型。然后，计算扰动后的原型的类别，这个类别是原型的类别乘以4。然后，将扰动后的原型和类别都移动到设备上，计算模型在扰动后的原型上的输出，然后计算输出和类别之间的交叉熵损失。

        最后，返回分类损失、原型增强损失和知识蒸馏损失的加权和。这里的权重是可以调节的超参数，可以根据实际情况进行调整。

        总的来说，这个方法的主要目的是计算模型的损失，包括分类损失、知识蒸馏损失和原型增强损失。这三种损失各有其特点，分类损失关注模型的预测能力，知识蒸馏损失关注模型的特征保持一致性，原型增强损失关注模型对原型的鲁棒性。
        """
        #暂存模型对imgs的输出
        output = self.model(imgs)
        output, target = output.to(self.device), target.to(self.device)
        #分类损失,计算方法为交叉熵损失，output除以温度参数，提高模型的泛化能力
        loss_cls = nn.CrossEntropyLoss()(output / self.args.temp,
                                            target.long())
        if self.old_model is None:
            return loss_cls
        else:
            feature = self.model.feature(imgs)
            feature_old = self.old_model.feature(imgs)
            #知识蒸馏损失，计算新旧模型的张量距离
            loss_kd = torch.dist(feature, feature_old, 2)

            proto_aug = []
            proto_aug_label = []
            index = list(range(old_class))
            #对原型进行随机扰动
            for _ in range(self.args.batch_size):
                np.random.shuffle(index)
                if len(self.prototype) <= index[0]:

                    temp = np.random.normal(0, 1, 512) * self.radius
                else:
                    temp = self.prototype[index[0]] + np.random.normal(
                        0, 1, 512) * self.radius
                proto_aug.append(temp)
                if len(self.class_label) <= index[0]:
                    proto_aug_label.append(0)
                else:
                    proto_aug_label.append(4 * self.class_label[index[0]])

            proto_aug = torch.from_numpy(np.float32(
                np.asarray(proto_aug))).float().to(self.device)
            proto_aug_label = torch.from_numpy(
                np.asarray(proto_aug_label)).to(self.device)
            soft_feat_aug = self.model.fc(proto_aug)
            #原型增强损失
            loss_protoAug = nn.CrossEntropyLoss()(
                (soft_feat_aug / self.args.temp), proto_aug_label.long())
            
            loss+=loss_cls + self.args.kd_weight * loss_kd + self.args.protoAug_weight * loss_protoAug
        if loss_fun_name == 'fedknow' and epoch>0:
            # optimizer=torch.optim.Adam(self.model.parameters(),
            #                    lr=self.learning_rate,
            #                    weight_decay=2e-4)
            loss_cut=0
            # model_outputs = self.model.forward(imgs,self.task_id)
            model_outputs = self.model.sm(imgs,self.task_id)
            for t in range(max(0,current_task-1)):
                
                begin, end = self.compute_offsets(t, self.numclass)
                
                model_output=model_outputs[:, begin:end]
                temppackmodel = copy.deepcopy(self.pack_models[t]).to(self.device)

                with torch.no_grad():
                    # pack_output = temppackmodel.forward(imgs, t)
                    pack_output = temppackmodel.sm(imgs, t)

                if pack_output.shape==model_output.shape:
                    # memoryloss = nn.CrossEntropyLoss()(model_output.reshape(-1), pack_output.reshape(-1))
                    memoryloss=torch.dist(model_output.reshape(-1), pack_output.reshape(-1),2)
                    if self.args.testmode==True:
                        print('t:',t)
                        print('memory loss:',memoryloss)
                    loss_cut += memoryloss
                else:
                    print(self.model.fc.weight.shape)
                    print(self.packmodel.fc.weight.shape)
                    print('begin:{},end:{}'.format(begin,end))
                    print('pack_output:{},model_output:{}'.format(pack_output.shape,model_output.shape))
                    
                del temppackmodel
            loss_cut/=(self.task_id)
            loss+=loss_cut*self.args.cut_weight
            if self.args.testmode:
                print(loss_cls,'\n',loss_kd*self.args.kd_weight,'\n',loss_protoAug*self.args.protoAug_weight,'\n',loss_cut*self.args.cut_weight)

        if self.args.testmode:
            if self.args.loss_fun_name!='fedknow':
                print(loss_cls,'\n',loss_kd*self.args.kd_weight,'\n',loss_protoAug*self.args.protoAug_weight,'\n')

        return loss
        

    def afterTrain(self):
        path = self.args.save_path + self.file_name + '/'
        if not os.path.isdir(path):
            os.makedirs(path)
        if self.numclass == self.args.fg_nc:#第一个任务
            self.history_accuracies=self._test(self.test_loader)
        else:
            self.history_accuracies=self._test(self.test_loader)
            if self.task_id<self.args.task_num:
                self.pack.apply_eval_mask(task_idx=self.task_id, model=self.packmodel.feature)
            self.packmodel.eval()

        self.numclass += self.task_size
        filename = path + '%d_model.pkl' % (self.numclass - self.task_size)
        torch.save(self.model, filename)
        self.old_model = torch.load(filename)
        self.old_model.to(self.device)
        self.old_model.eval()
        

    def protoSave(self, model, loader, current_task):
        """
        这段代码定义了一个名为 protoSave 的方法，用于保存模型的原型。这个方法接收三个参数：模型、数据加载器和当前任务的索引。
        首先，代码创建了三个空列表prototype、radius和class_label，用于存储每个类别的原型、半径和标签。

        然后，代码遍历了所有的唯一标签（labels_set）。对于每个标签，它首先找到所有属于该类别的样本（np.where(item == labels)[0]），然后计算这些样本的特征的平均值，作为该类别的原型，并将其添加到prototype列表中。

        如果当前任务是第一个任务（current_task == 0），那么代码还会计算该类别的半径。半径是通过计算特征的协方差矩阵的迹，然后除以特征的维度得到的。这个半径被添加到radius列表中。

        在计算完所有类别的原型和半径后，如果当前任务是第一个任务，那么代码会将radius列表中的所有值取平方根，然后计算平均值，作为最终的半径，并将其保存为类的属性。同时，prototype和class_label列表也被保存为类的属性。

        如果当前任务不是第一个任务，那么代码会将新计算的原型和类别标签添加到已有的原型和类别标签中。这是通过np.concatenate函数实现的，该函数可以将两个数组沿指定的轴连接起来。
        """

        features = []
        labels = []
        model.eval()
        with torch.no_grad():
            for i, (indexs, images, target) in enumerate(loader):
                feature = model.feature(images.to(self.device))
                #TODO 几个reshape整不会了
                if feature.shape[0] == self.args.batch_size:
                    labels.append(target.numpy())
                    features.append(feature.cpu().numpy())
        labels_set = np.unique(labels)
        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
        features = np.array(features)
        features = np.reshape(
            features,
            (features.shape[0] * features.shape[1], features.shape[2]))
        feature_dim = features.shape[1]

        prototype = []  #原型
        radius = []  #原型的半径，通过原型生成时会用到
        class_label = []  #类标签
        for item in labels_set:
            index = np.where(item == labels)[0]
            class_label.append(item)
            feature_classwise = features[index]
            prototype.append(np.mean(feature_classwise, axis=0))
            if current_task == 0:
                cov = np.cov(feature_classwise.T)
                radius.append(np.trace(cov) / feature_dim)

        if current_task == 0:
            self.radius = np.sqrt(np.mean(radius))
            self.prototype = prototype
            self.class_label = class_label
            print(self.radius)
        else:
            self.prototype = np.concatenate((prototype, self.prototype),
                                            axis=0)
            self.class_label = np.concatenate((class_label, self.class_label),
                                              axis=0)
            
#fedknow-fisher
    def compute_offsets(self,task, nc_per_task, is_cifar=True):
        """
            Compute offsets for cifar to determine which
            outputs to select for a given task.
        """
        if task==0:
            offset1 = 0
            offset2 = self.args.fg_nc*4
        else:
            offset1 = self.numclass*4-self.task_size*4*task
            offset2 = offset1+self.task_size*4
        return offset1, offset2

    # def fisher_matrix_diag(self,t):
    #     # Init
    #     fisher = {}
    #     _model=self.model

    #     for n, p in _model.feature.named_parameters():
    #         fisher[n] = 0 * p.data
    #     # Compute
    #     _model.train()
    #     criterion = torch.nn.CrossEntropyLoss()
    #     offset1, offset2 = self.compute_offsets(t, self.task_size)
    #     all_num = 0
    #     for step, (indexs, images, target) in enumerate(self.train_loader):
    #         images, target = images.to(self.device), target.to(self.device)
    #         target = (target - 10 * t)
    #         all_num += images.shape[0]
    #         # Forward and backward
    #         _model.zero_grad()
    #         outputs = _model.forward(images, t)[:, offset1: offset2]
    #         loss = criterion(outputs, target.long())
    #         loss.backward(retain_graph=True)
    #         # Get gradients
    #         for n, p in _model.feature.named_parameters():
    #             if p.grad is not None:
    #                 fisher[n] += images.shape[0] * p.grad.data.pow(2)
    #     # Mean
    #     with torch.no_grad():
    #         for n, _ in _model.feature.named_parameters():
    #             fisher[n] = fisher[n] / all_num
    #     return fisher
    def criterion(self, t):
        # Regularization for all previous tasks
        loss_reg = 0
        if t > 0:
            for (name, param), (_, param_old) in zip(self.model.feature.named_parameters(), self.old_model.feature.named_parameters()):
                loss_reg += torch.sum(self.fisher[name] * (param_old - param).pow(2)) / 2
        return loss_reg