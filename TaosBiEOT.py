import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


def main():
    # TD:
    d = 8  # dimension
    n_epoch = 100  # number of epoch each iteration
    eps = 0.5  # set regularization term (epsilon)
    n = 2048  # sample size
    K = 32  # neuron size
    lr = 1e-3  # learning rate
    bs = 64  # batch size


    X = torch.from_numpy(
        np.random.uniform(-1 / np.sqrt(d), 1 / np.sqrt(d), (n, d))).float()  # uniform [-1/sqrt(d),1/sqrt(d)]^d
    Y = torch.from_numpy(np.random.uniform(-1 / np.sqrt(d), 1 / np.sqrt(d), (n, d))).float()



    f = Net_EOT(d, K)
    optim_f = torch.optim.Adam(list(f.parameters()), lr=lr)


    for epoch in range(n_epoch):
        x_b = DataLoader(X, batch_size=bs, shuffle=True)
        y_b = DataLoader(Y, batch_size=bs, shuffle=True)
        for x, y in zip(x_b, y_b):
            optim_f.zero_grad()
            pred_x = f(x)

            x_norm = torch.norm(x, dim=-1) ** 2
            y_norm = torch.norm(y, dim=-1) ** 2
            cost_ = -4 * x_norm[:, None] * y_norm[None, :]

            cost = torch.norm(x[:,None] - y[None,:], dim=-1) ** 2

            loss = semidual_loss(pred_x, cost, eps)
            loss.backward()
            optim_f.step()
        print(f'loss: {loss.item()}')
    f.eval()




def semidual_loss(pred_x,cost,eps):
    max_x = pred_x.max() # stablized
    ret = torch.mean(pred_x) - eps* torch.mean(torch.log(torch.mean(torch.exp((pred_x-cost-max_x)/eps),dim=0))) - max_x + eps
    # ret = torch.mean(pred_x) - eps* torch.mean(torch.log(torch.mean(torch.exp((pred_x-cost-max_x)/eps),dim=0))) +eps
    loss = - ret  # maximize
    return loss


class Net_EOT(nn.Module):
    def __init__(self,dim,K):
        super(Net_EOT, self).__init__()

        self.fc1 = nn.Linear(dim, K)
        self.fc2 = nn.Linear(K, 1)

    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        x2 = self.fc2(x1)
        return x2







if __name__ == '__main__':
    main()