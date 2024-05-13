import numpy as np
import torch.nn as nn
import torch
import wandb
from torch.utils.data import DataLoader
import torch.nn.functional as F
from timeit import default_timer as timer
from utils.data_fns import QuadCost
import pickle
import os


class MOT_NE_alg():
    def __init__(self, params, device):
        self.models = []
        self.k = params['k']
        self.num_epochs = params['epochs']
        self.batch_size = params['batch_size']
        self.eps = params['eps']
        for i in range(params['k']):
            # model = NE_mot_model(dim=params['dims'][i], hidden_dim=params['hidden_dim'])
            # model.to(device)
            ### Tao's
            d = params['dims'][i]
            model = Net_EOT(dim=d, K=min(6*d,80), deeper=True).to(device)
            ###
            self.models.append(model)
        self.cost = params['cost']
        self.opt = [torch.optim.Adam(list(self.models[i].parameters()), lr=params['lr']) for i in range(self.k)]
        self.device = device
        self.cost_graph = params['cost_graph']
        self.using_wandb = params['using_wandb']
        self.params = params


    def train_mot(self, X):
        x_b = DataLoader(X, batch_size=self.batch_size, shuffle=True)
        self.tot_loss = []
        self.times = []
        for epoch in range(self.num_epochs):
            l = []
            t0 = timer()
            for i, data in enumerate(x_b):
                self.zero_grad_models()
                # data.to(self.device)
                for k_ind in range(self.k):
                    # k-margin loss
                    phi = [self.models[i](data[:,:,i]) for i in range(self.k)]  # each phi[i] should have shape (b,1)
                    e_term = self.calc_exp_term(phi, data)
                    loss = -(sum(phi).mean() - self.eps*e_term)
                    ####
                    # bimargin loss (for debugging)
                    # cost = self.calc_cost(data)
                    # loss = dual_loss(pred_x = phi[0], pred_y = phi[1], cost=cost, eps=self.eps)
                    ####
                    loss.backward()
                    self.opt[k_ind].step()
                    l.append(loss.item())
            l = np.mean(l)
            self.tot_loss.append(-l+ self.eps)
            epoch_time = timer()-t0
            self.times.append(epoch_time)
            print(f'finished epoch {epoch}, loss={-l+ self.eps:.5f}, took {epoch_time:.2f} seconds')
            # print(f'finished epoch {epoch}, loss={l / i}')
        self.models_to_eval

    def save_results(self):
        tot_loss = np.mean(self.tot_loss[-10:])
        avg_time = np.mean(self.times)
        data_to_save = {
            'avg_loss': tot_loss,
            'avg_time': avg_time,
            'tot_loss': self.tot_loss,
            'times': self.times,
            'params': self.params
            }
        # Save path
        path = os.path.join(self.params.figDir, 'results.pkl')

        # Saving the data using pickle
        with open(path, 'wb') as file:
            pickle.dump(data_to_save, file)

        if self.using_wandb:
            wandb.log({'tot_loss': tot_loss, 'avg_time': avg_time})


        print(f'Finished run, loss is {tot_loss:.5f}, average epoch time is {avg_time:.3f} seconds')


    def calc_exp_term(self, phi, x):
        # TD - IMPLEMENT MORE MEMORY EFFICIENT CALCULATION!!! DIVIDE LOSS!!

        # calc loss tensor
        c = self.calc_cost(x)

        if self.cost_graph == 'circle':
            # calc mapping:
            reshaped_term = []
            reshaped_c = []
            for index, vec in enumerate(phi):
                # Create a shape of length k with 1s except at the index position
                shape = [1] * self.k
                shape[index] = -1
                reshaped_c.append(c[index].reshape(shape))
                reshaped_term.append(vec.reshape(shape))
            reshaped_term = sum(reshaped_term)
            c = sum(reshaped_c)
        elif self.cost_graph == 'full':
            reshaped_term = []
            for index, vec in enumerate(phi):
                # Create a shape of length k with 1s except at the index position
                shape = [1] * self.k
                shape[index] = -1
                # shape[-index] = -1
                reshaped_term.append(vec.reshape(shape))
            reshaped_term = sum(reshaped_term)
            # reshaped_term = phi[0][None, :] + phi[1][:, None]



        return torch.mean(torch.exp((reshaped_term-c)/self.eps))


    def calc_cost(self, data):
        """
        calculates the cost over bacthed data
        :param data:
        :return:
        """
        if self.cost == 'quad':
            cost = QuadCost(data, mod=self.cost_graph)
        elif self.cost == 'quad_gw':
            # IMPLEMENT - cost = QuadCostGW(data, self.matrices)
            pass
        elif self.cost == 'ip_gw':
            # IMPLEMENT - cost = IPCostGW(data, self.matrices)
            pass

        # NOW - BROADCAST!!
        return cost

    def zero_grad_models(self):
        for opt in self.opt:
            opt.zero_grad()

    def models_to_eval(self):
        for model in self.models:
            model.eval()

class Net_EOT(nn.Module):
    def __init__(self,dim,K,deeper=False):
        super(Net_EOT, self).__init__()
        self.deeper=deeper
        if deeper:
            self.fc1 = nn.Linear(dim, 10 * K)
            self.fc2 = nn.Linear(10 * K, 10 * K)
            self.fc3 = nn.Linear(10 * K, 1)
        else:
            self.fc1 = nn.Linear(dim, K)
            self.fc2 = nn.Linear(K, 1)

    def forward(self, x):
        if self.deeper:
            x1 = F.relu(self.fc1(x))
            x11 = F.relu(self.fc2(x1))
            x2 = self.fc3(x11)
        else:
            x1 = F.relu(self.fc1(x))
            x2 = self.fc2(x1)
        return x2


class NE_mot_model(nn.Module):
    def __init__(self, dim, hidden_dim=32):
        super(NE_mot_model, self).__init__()
        self.dim = dim

        # Taos's N-GW model
        self.fc1 = nn.Linear(dim, self.dim)
        self.fc2 = nn.Linear(self.dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
