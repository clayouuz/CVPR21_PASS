import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import os
import math
import time
import argparse
# import resource
import pandas as pd
import numpy as np
import sklearn.metrics
from scipy import stats
from PIL import Image

from PASS import protoAugSSL
from ResNet import resnet18_cbam
from myNetwork import network
from iCIFAR100 import iCIFAR100

parser = argparse.ArgumentParser(description='Prototype Augmentation and Self-Supervision for Incremental Learning')
parser.add_argument('--epochs', default=101, type=int, help='Total number of epochs to run')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
parser.add_argument('--print_freq', default=10, type=int, help='print frequency (default: 10)')
parser.add_argument('--data_name', default='cifar100', type=str, help='Dataset name to use')
parser.add_argument('--total_nc', default=100, type=int, help='class number for the dataset')
parser.add_argument('--fg_nc', default=50, type=int, help='the number of classes in first task')
parser.add_argument('--task_num', default=10, type=int, help='the number of incremental steps')
parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--protoAug_weight', default=10.0, type=float, help='protoAug loss weight')
parser.add_argument('--kd_weight', default=10.0, type=float, help='knowledge distillation loss weight')
parser.add_argument('--temp', default=0.1, type=float, help='trianing time temperature')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')
parser.add_argument('--save_path', default='model_saved_check/', type=str, help='save files directory')
parser.add_argument('--loss_fun_name',default='pass',type=str,help='loss function name')
parser.add_argument('--drop_penalty_weight', default=0.001, type=float, help='drop penalty weight')
parser.add_argument('--testmode', default=False, type=bool, help='if True, train on only 1000 samples')
parser.add_argument('--proto_gen', default=False, type=bool, help='generate prototype in use')
parser.add_argument('--fisher_loss', default=False, type=bool, help='use fisher loss')


args = parser.parse_args()
print(args)

def map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))

# 用于设置数据，将数据集中的类别随机打乱
def setup_data(test_targets, shuffle, seed):
    order = [i for i in range(len(np.unique(test_targets)))]
    if shuffle: 
        np.random.seed(seed)
        order = np.random.permutation(len(order)).tolist()
    else:
        order = range(len(order))
    class_order = order
    print(100 * '#')
    print(class_order)
    return map_new_class_index(test_targets, class_order)

def main():
    """
    Main function that executes the training and testing process.

    This function performs the following steps:
    1. Sets up the device for training (either GPU or CPU).
    2. Calculates the number of classes in each incremental step.
    3. Sets up the file name for saving the model.
    4. Initializes the feature extractor.
    5. Creates an instance of the `protoAugSSL` class.
    6. Sets up the data for training.
    7. Iterates over the number of tasks and performs training for each task.
    8. Performs testing for each task.
    9. Prints the accuracy results.

    Note: This function assumes that the necessary modules and variables are imported and defined.

    Args:
        None

    Returns:
        None
    """
    #set up device
    cuda_index = 'cuda:' + args.gpu
    device = torch.device(cuda_index if torch.cuda.is_available() else "cpu")

    #每步增量学习的类别数。task_size = (总类别数 - 第一步类别数) / 总步数
    task_size = int((args.total_nc - args.fg_nc) / args.task_num)  
    file_name = args.data_name + '_' + str(args.fg_nc) + '_' + str(args.task_num) + '+' + str(task_size) 
    #resnet18_cbam是一个ResNet18网络,历史悠久的公开方案，用来提取特征
    feature_extractor = resnet18_cbam()

    #定义模型，详见PASS.py
    model = protoAugSSL(args, file_name, feature_extractor, task_size, device)

    class_set = list(range(args.total_nc))
    model.setup_data(shuffle=True, seed=1993)
    # for循环遍历所有任务，oldclass存入上一个任务的类别数，然后训练
    # 初始化时oldclass=0，即第一个任务的类别数为0
    for i in range(args.task_num+1):
        if i == 0:
            old_class = 0
        else:
            old_class = len(class_set[:args.fg_nc + (i - 1) * task_size])
        model.beforeTrain(i)
        model.train(i, old_class=old_class)
        model.afterTrain()

    # Rest of the code...


if __name__ == "__main__":
    time_start = time.time()
    main()
    time_end = time.time()
    print('time cost', time_end - time_start, 's')
