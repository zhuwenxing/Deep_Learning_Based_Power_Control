import numpy as np 
import torch as t
import torch.nn as nn



class ConvLayer(nn.Module):
    def __init__(self, in_channels,out_channels,kernel_size,stride):
        super(ConvLayer,self).__init__()
        zero_padding = int(np.floor(kernel_size / 2))
        self.zero_pad = nn.ZeroPad2d(zero_padding)
        self.conv2d = nn.Conv2d(in_channels,out_channels,kernel_size,stride)
        self.relu = nn.ReLU()
    def forward(self,x):
        out = self.zero_pad(x)
        out = self.conv2d(out)
        out = self.relu(out)
        return out


class inter_conv_net(nn.Module):
# 有一个很重要的问题： 在tensorflow中，卷积操作的输入是 batch, in_height, in_width, channel

# 但是在Pytorch中，输入是batch, channel,in_height, in_width

    def __init__(self):
        super(inter_conv_net, self).__init__()
        self.conv_0 = ConvLayer(5,16,kernel_size = 1,stride = 1)
        self.conv_1 = ConvLayer(16,16,kernel_size = 1,stride = 1)
        self.conv_2 = ConvLayer(16,6,kernel_size=1,stride = 1)
         
    def forward(self,inter):


        h1 = self.conv_0(inter)
        h2 = self.conv_1(h1)
        h3 = self.conv_2(h2)
        fea_sum = t.sum(h3,dim=3) # reduce是在用户维度
        fea_max = t.max(h3,dim=3)[0] # 这个地方 max 返回的是一个元祖，包括张量和index
        fea = t.cat((fea_sum,fea_max),dim=1) # cat是在通道维度

        return fea
# 所以问题还是在卷积层这一块，需要将通道维度先放在1维的位置，进行完卷积后，在将通道维度放回最后一维。
class fully_connected_net(nn.Module):
    def __init__(self):
        super(fully_connected_net, self).__init__()
        self.fc1 = nn.Linear(15,32) # 可以将三维数据输入到fc中
        self.fc2 = nn.Linear(32,16)
        self.fc3 = nn.Linear(16,1)
        self.relu = nn.ReLU()
    def forward(self,feat):
        hidden = self.relu(self.fc1(feat))
        hidden = self.relu(self.fc2(hidden))
        out = self.fc3(hidden)

        return out


class IGCNet(nn.Module):
    def __init__(self,L,layer_num):
        super(IGCNet,self).__init__()
        self.L = L
        self.layer_num = layer_num
        self.inter_conv_net = inter_conv_net()
        self.fully_connected_net = fully_connected_net()
        self.sigmoid = nn.Sigmoid()

    def forward(self,Xinterf,Xintert,Xdiag,Xdiag_o,intensity,w_alpha): # 这个地方的输入比较多
        """
        Xinterf: None,L,L,1
        Xintert: None,L,L,1
        Xdiag: None,L,1
        Xdiag_o: None,L,L,1
        intensity: None,L,1
        w_alpha: None,L,1

        """
        
        
        # 先预处理

        intens = intensity
        all_one = t.ones((1,self.L))
        intens2 = t.mul(intens,all_one)
        w2 = t.mul(w_alpha,all_one) # w_alpha 少了一维！
        intens2 = intens2.permute(0,2,1)
        intens2 = t.reshape(intens2,(-1,self.L,self.L,1))
        w2 = w2.permute(0,2,1)
        w2 = t.reshape(w2,(-1,self.L,self.L,1))
        xf = t.cat((Xinterf,intens2,Xintert,Xdiag_o,w2),dim=3)

        # 预处理结束  应该主要是对intens和w2 进行了广播扩维
        for ii in range(self.layer_num):
            xf = xf.permute(0,3,1,2)
            fea1 = self.inter_conv_net(xf) # 这里的输出只有三个维度
            fea1 = fea1.permute(0,2,1)
            fea = t.cat((Xdiag,intens,fea1,w_alpha),dim=2)
            out = self.fully_connected_net(fea)
            pred = self.sigmoid(out)
            # 对下一层网络的输入进行更新 主要也就是对intens的维度修改 和再次cat
            intens = pred  # intens是一直在变化的
            intens2 = t.mul(intens,t.ones((1,self.L)))
            intens2 = intens2.permute(0,2,1)
            intens2 = t.reshape(intens2,(-1,self.L,self.L,1))
            xf = t.cat((Xinterf,intens2,Xintert,Xdiag_o,w2),dim=3)

        return pred


# if __name__ == "__main__":
#     net = IGCNet(10,5)

#     # 构造输入数据
#     Xinterf = t.randn(1,10,10,1)
#     Xintert = t.randn(1,10,10,1)
#     Xdiag = t.randn(1,10,1)
#     Xdiag_o = t.randn(1,10,10,1)
#     intensity = t.randn(1,10,1)
#     w_alpha = t.randn(1,10,1)
#     pred = net(Xinterf,Xintert,Xdiag,Xdiag_o,intensity,w_alpha)
#     print(pred.shape)
