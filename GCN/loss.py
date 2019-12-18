
import torch as t
import torch.nn as nn

class RateLoss(nn.Module):
    """
    abs_H: batch_size,K,K (批次大小，用户维度，基站维度) 所以在forward中要对abs_H转置-》（批次，基站维度，用户维度）
    power：batch_size,K 在forward中会reshape为 batch_size,K,1
    """

    def __init__(self,K,var):
        super(RateLoss, self).__init__()
        self.K = K
        self.var = var
    
    def forward(self, abs_H, power):
        power = t.reshape(power, (-1, self.K, 1)) # 可以兼容两种形式
        abs_H = abs_H.permute(0,2,1)
        abs_H_2 = t.pow(abs_H, 2)  # 平方后的abs_H
        rx_power = t.mul(abs_H_2, power)
        mask = t.eye(self.K)
        valid_rx_power = t.sum(t.mul(rx_power, mask), 1)
        interference = t.sum(t.mul(rx_power, 1 - mask), 1) + self.var
        rate = t.log2(1 + t.div(valid_rx_power, interference))
        sum_rate = t.mean(t.sum(rate, 1))
        loss = t.neg(sum_rate)        

        return loss



# Test

if __name__ == "__main__":
    power = t.ones(2, 20)
    abs_H = t.ones(2, 20, 20)
    
    criterion = RateLoss(20, 1)
    loss = criterion(abs_H, power)
    print(loss.item())
