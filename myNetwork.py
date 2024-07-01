import torch.nn as nn
import torch


class network(nn.Module):

    def __init__(self, numclass, feature_extractor):
        super(network, self).__init__()  #调用父类的初始化函数
        self.feature = feature_extractor
        self.fc = nn.Linear(512, numclass,
                            bias=True)  #一个全连接层，输入维度512，输出维度numclass

    def forward(self, input):  #前向传播
        x = self.feature(input)
        x = self.fc(x)
        return x

    def Incremental_learning(self, numclass):  #增量学习方法
        """
        用一个numclass维的全连接层替换原来的全连接层,前面的权重和偏置不变
        """
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features

        #创建新层，只保留前out_feature个权重和偏置
        self.fc = nn.Linear(in_feature, numclass, bias=True)
        self.fc.weight.data[:out_feature] = weight[:out_feature]
        self.fc.bias.data[:out_feature] = bias[:out_feature]

    def feature_extractor(self, inputs):
        return self.feature(inputs)
