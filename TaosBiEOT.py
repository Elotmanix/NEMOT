import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from ot.bregman import sinkhorn_knopp
import os
from models import Net_EOT
from data_fns import gen_data

def main():
    # TD:
    d = 5  # dimension
    n_epoch = 200  # number of epoch each iteration
    eps = 0.5  # set regularization term (epsilon)
    n = 5000  # sample size
    K = min(6*d,80)  # neuron size, tao had 32 fixed
    lr = 5e-4  # learning rate - 1e-3 worked best for Taos semidual original
    bs = 64  # batch size

    two_fns=True
    solve_classic = False
    fixed_vol = True

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    print(f'd={d},n={n},eps={eps},lr={lr},bs={bs},two_fns={two_fns}, fixed_vol={fixed_vol}')

    if fixed_vol:
        X = torch.from_numpy(
            np.random.uniform(-1 / np.sqrt(d), 1 / np.sqrt(d), (n, d))).float().to(device)  # uniform [-1/sqrt(d),1/sqrt(d)]^d
        Y = torch.from_numpy(np.random.uniform(-1 / np.sqrt(d), 1 / np.sqrt(d), (n, d))).float().to(device)
    else:
        params = {
            'k': 2,
            'alg': 'ne_mot',
            'dims': [d,d,d,d],
            'data_dist': 'uniform',
            'n': n
        }
        data_x = gen_data(params)
        data_x = data_x.to(device)
        X = data_x[:,:,0]
        Y = data_x[:, :, 1]
        # prev implementation:
        # X = torch.from_numpy(
        #     np.random.uniform(-0.5, 0.5, (n, d))).float().to(
        #     device)  # uniform [-1/sqrt(d),1/sqrt(d)]^d
        # Y = torch.from_numpy(np.random.uniform(-0.5, 0.5, (n, d))).float().to(device)

    if solve_classic:
        log,gamma = solve_unif_sinkhorn(X,Y,eps,n)



    if two_fns:
        deeper=True
        f1 = Net_EOT(d, K, deeper).to(device)
        f2 = Net_EOT(d, K, deeper).to(device)
        # optim_f = torch.optim.Adam(list(f1.parameters())+list(f2.parameters()), lr=lr)
        optim_f1 = torch.optim.Adam(list(f1.parameters()), lr=lr)
        optim_f2 = torch.optim.Adam(list(f2.parameters()), lr=lr)
    else:
        f = Net_EOT(d, K).to(device)
        optim_f = torch.optim.Adam(list(f.parameters()), lr=lr)

    x_b = DataLoader(X, batch_size=bs, shuffle=True)
    y_b = DataLoader(Y, batch_size=bs, shuffle=True)
    epoch_losses = []
    for epoch in range(n_epoch):
        epoch_loss = []
        for x, y in zip(x_b, y_b):
            if two_fns:
                for k in range(2):
                    if k:
                        optim_f1.zero_grad()
                    else:
                        optim_f2.zero_grad()
                    pred_x = f1(x)
                    pred_y = f2(y)
                    cost = torch.norm(x[:, None] - y[None, :], dim=-1) ** 2
                    loss = dual_loss(pred_x, pred_y, cost, eps)
                    loss.backward()
                    if k:
                        optim_f1.step()
                    else:
                        optim_f2.step()
                epoch_loss.append(-loss.item()+eps)
            else:
                optim_f.zero_grad()
                pred_x = f(x)
                cost = torch.norm(x[:,None] - y[None,:], dim=-1) ** 2
                loss = semidual_loss(pred_x, cost, eps)
                loss.backward()
                optim_f.step()
                epoch_loss.append(-loss.item())
        epoch_losses.append(epoch_loss)
        epoch_loss = np.mean(epoch_loss)
        print(f'loss: {epoch_loss}')
    print(f'final 10 epochs average to {np.mean(epoch_losses[-10:])}')
    # f.eval()



def solve_unif_sinkhorn(X,Y,eps,n):
    a = (1 / n) * np.ones(n)
    b = (1 / n) * np.ones(n)

    C = np.linalg.norm(X[:,None] - Y[None,:], axis=-1) ** 2
    p, log = sinkhorn_knopp(a=a,b=b,M=C,reg=eps, verbose=True, log=True)
    return p,log




def semidual_loss(pred_x,cost,eps):
    # max_x = pred_x.max() # stablized
    max_x = 0
    ret = torch.mean(pred_x) - eps* torch.mean(torch.log(torch.mean(torch.exp((pred_x-cost-max_x)/eps),dim=0))) - max_x
    # ret = torch.mean(pred_x) - eps* torch.mean(torch.log(torch.mean(torch.exp((pred_x-cost-max_x)/eps),dim=0))) +eps
    loss = - ret  # maximize
    return loss

def dual_loss(pred_x, pred_y, cost,eps):
    # max_x = pred_x.max() # stablized
    max_x = 0
    ret = torch.mean(pred_x) + torch.mean(pred_y) - eps*torch.mean(torch.exp((pred_x[None,:]+pred_y[:,None]-cost)/eps))
    # ret = torch.mean(pred_x) - eps* torch.mean(torch.log(torch.mean(torch.exp((pred_x-cost-max_x)/eps),dim=0))) +eps
    loss = - ret  # maximize
    return loss










if __name__ == '__main__':
    main()