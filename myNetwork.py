import torch.nn as nn
import torch


class network(nn.Module):

    def __init__(self, numclass, feature_extractor):
        super(network, self).__init__()  #调用父类的初始化函数
        self.feature = feature_extractor
        self.numclass = numclass
        self.fc = nn.Linear(512, numclass,
                            bias=True)  #一个全连接层，输入维度512，输出维度numclass

        self.weight_keys = [['feature_net.conv1.weight'],
                            ['feature_net.conv1.bias'],
                            ['feature_net.conv2.weight'],
                            ['feature_net.conv2.bias'],
                            ['feature_net.conv3.weight'],
                            ['feature_net.conv3.bias'],
                            ['feature_net.conv4.weight'],
                            ['feature_net.conv4.bias'],
                            ['feature_net.conv5.weight'],
                            ['feature_net.conv5.bias'],
                            ['feature_net.conv6.weight'],
                            ['feature_net.conv6.bias'],
                            ['feature_net.fc1.weight'],
                            ['feature_net.fc1.bias'], ['last.weight'],
                            ['last.bias']]

    # def forward(self, input):  #前向传播
    #     x = self.feature(input)
    #     x = self.fc(x)
    #     return x
    def forward(self, x, t=-1, pre=False, is_con=True, avg_act=False):
        if t==-1:
            h = self.feature(x)
            output = self.fc(h)
            return output
        
        h = self.feature(x)
        output = self.fc(h)
        if is_con:
            # make sure we predict classes within the current task
            if pre:
                offset1 = 0
                offset2 = int(t * 10)#t*class_num_per_task
            else:
                offset1 = int(t * 10)
                offset2 = int((t + 1) * 10)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.numclass:
                output[:, offset2:self.numclass].data.fill_(-10e10)
        return output

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


class RepTail(nn.Module):

    def __init__(self,
                 inputsize,
                 output=100,
                 nc_per_task=10,
                 feature_extractor=None,
                 numclass=100):
        super().__init__()
        self.feature_net = feature_extractor
        self.last = nn.Linear(512, numclass,
                              bias=True)  #一个全连接层，输入维度512，输出维度numclass
        self.nc_per_task = nc_per_task
        self.n_outputs = output
        self.weight_keys = [['feature_net.conv1.weight'],
                            ['feature_net.conv1.bias'],
                            ['feature_net.conv2.weight'],
                            ['feature_net.conv2.bias'],
                            ['feature_net.conv3.weight'],
                            ['feature_net.conv3.bias'],
                            ['feature_net.conv4.weight'],
                            ['feature_net.conv4.bias'],
                            ['feature_net.conv5.weight'],
                            ['feature_net.conv5.bias'],
                            ['feature_net.conv6.weight'],
                            ['feature_net.conv6.bias'],
                            ['feature_net.fc1.weight'],
                            ['feature_net.fc1.bias'], ['last.weight'],
                            ['last.bias']]

    def forward(self, x, t, pre=False, is_con=True, avg_act=False):
        h = self.feature_net(x, avg_act)
        output = self.last(h)
        if is_con:
            # make sure we predict classes within the current task
            if pre:
                offset1 = 0
                offset2 = int(t * self.nc_per_task)
            else:
                offset1 = int(t * self.nc_per_task)
                offset2 = int((t + 1) * self.nc_per_task)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output
