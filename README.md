
# PyTorch Version: implement of power control with different neural networks

## Motivation
个人认为PyTorch框架下，写出来的代码会更有逻辑性，更易于理解。但是在我创建的  [Paper-with-Code-of-Wireless-communication-Based-on-DL](https://github.com/IIT-Lab/Paper-with-Code-of-Wireless-communication-Based-on-DL) 中，我发现大部分的代码都还是TensorFlow版本的(少量Keras)。在TF转为动态图之前的版本，基于TF的代码会比较难调试。刚入门深度学习时，学的也是TF和Kreas。和大部分人的感受一样，写TF的代码会感觉和Python有很大的割裂感，而Keras则是不够灵活。当时还是Torch，基于lua，所以还要重新学一门新语言，也就没有学。当FaceBook推出PyTorch,得到了很大的好评，所以就立马入坑了。

如果以后有时间，会尝试将上面那个库中感兴趣一些代码改写为基于PyTorch的版本。

## Introduction

这个库只关注基于深度学习的功率控制。其实这个方向有点... 最开始就是使用DNN去拟合一个传统的功率控制算法WMMSE。后面就是CNN LSTM GCN网络的应用。当然还有一个改进就是将之前的监督式的方式，改为优化目标导向的无监督方式。

参考文献：

MLP：

    * 论文：[Learning to Optimize: Training Deep Neural Networks for Interference Management](https://ieeexplore.ieee.org/document/8444648)
    * 代码：[SPAWC2017](https://github.com/Haoran-S/SPAWC2017)

CNN:

    * 论文：[Deep Power Control Transmit Power Control Scheme Based on Convolutional Neural Network](https://ieeexplore.ieee.org/document/8335785)

LSTM:

    * 论文：[Application of deep learning for power control in the interference channel a RNN-based approach](https://dl.acm.org/citation.cfm?id=3355668)

GCN:

    * 论文：[A Graph Neural Network Approach for Scalable Wireless Power Control](https://arxiv.org/abs/1907.08487)
    * 代码：[Globecom2019](https://github.com/yshenaw/Globecom2019)


第一篇应该是开创性的工作，第二篇我自己觉得有意义的点不在于CNN的网络结构而是pre train和目标导向的损失函数，第四篇使用的图结构，在我实验的过程中看来，应该是效果最好的！

## Some Issues
* 碰到的第一个坑就是，不同的论文关于信道矩阵的定义方式不同，例如```H[i][j]```中，有些论文是```i：基站维度，j：用户维度```，有些论文则是反过来的
* 在计算sum rate的时候，用到了很多矩阵的运算，大脑不能很好的想象每一步矩阵运算后的维度变化以及每个element的计算方式（有时候是点乘，有时候是叉乘）。当数据的维度超过三维后，就容易晕。
* TF中张量竟然是将channl的维度放在最后一维....即shape为(batch_size,height,width,channel)
* 在PyTorch中的一个坑就是使用将numpy的array加载成DataLoader，因为numpy是float64，所以tensor也是float64。但是在PyTorch中tensor默认是float32,所以两者运算就会出错。


## Setup
个人建议使用virtualenvwrapper的包管理工具，可参考: https://virtualenvwrapper.readthedocs.io/en/latest/

同时安利一下VSCode编辑器。

requirement：

* torch==1.3.0+cpu
* numpy==1.17.3
* scipy==1.3.1

虽然可以使用requirements.txt安装。

目前将 [Globecom2019](https://github.com/yshenaw/Globecom2019) 这份TensorFlow的代码修改为了PyTorch版本。运行后结果和原始代码基本是一致的。其他网络只给出了网络结构的代码，后序会继续Update。其实这份代码的主要关注点就是network.py和loss.py的。