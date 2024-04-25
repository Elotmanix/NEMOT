import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F



"""
First test implementation of NMOT to check no problems with the idea itself
Code flow:
0. define simulation aprameters
1. define data 
2. define neural estimator nets (options - a)k different neural nets or b)weight sharing/one net with learned weigted output)
3. run on example
4. compare with Sinkhorn output (I need a Sinkhorn implementation anyways)
"""
def main():
    # TD:
    # args = argparse()
    # params = gen_params(args)

    params = gen_params()

    X = gen_data(params)

    if params['alg'] == 'ne_mot':
        MOT_agent = MOT_NE_alg(params)
        MOT_agent.train_mot(X)





########
def gen_params():
    params = {
        'batch_size': 128,
        'epochs': 2000,
        'lr': 1e-3,
        'n': 2048,
        'k': 3,
        'eps': 0.5,
        'cost': 'quad',  # options - quad, quad_gw, ip_gw
        'alg': 'ne_mot',        #options - ne_mot, sinkhorn_mot,ne_gw, sinkhorn_gw
        'hidden_dim': 32,
        'mod': 'mot',       #options - mot, mgw
        'seed': 42,
        'data_dist': 'uniform',
        'dims': [8,8,8,8,8,8,8,8],
    }
    # TD: ADJUST DIMS TO K
    return params


def gen_data(params):
    if params['data_dist'] == 'uniform':
        # generate k samples which are d-dimensional with n samples (from Taos's notebook)
        X = []
        for i in range(params['k']):
            X.append(np.random.uniform(-1/np.sqrt(params['dims'][i]),1/np.sqrt(params['dims'][i]),(params['n'],params['dims'][i])).astype(np.float32))
        X = np.stack(X, axis=-1)

        if params['alg'] != 'ne':
            MU = [(1 / params['n']) * np.ones(params['n'])]*params['k']
            return X, MU

        return X


def QuadCost(data):
    differences = [torch.norm(data[:, :, i] - data[:, :, i + 1], dim=1) for i in range(self.k - 1)]
    differences.append(torch.norm(data[:, :, -1] - data[:, :, 0], dim=1) ** 2)
    return differences

class MOT_NE_alg():
    def __init__(self, params):
        self.models = []
        self.k = params['k']
        self.num_epochs = params['epochs']
        self.batch_size = params['batch_size']
        self.eps = params['eps']
        for i in range(params['k']):
            self.models.append(NE_model(dim=params['dims'][i], hidden_dim=params['hidden_dim']))
        self.cost = params['cost']
        self.opt = [torch.optim.Adam(list(self.models[i].parameters()), lr=params['lr']) for i in range(self.k)]


    def train_mot(self, X):
        for epoch in range(self.num_epochs):
            x_b = DataLoader(X, batch_size=self.batch_size, shuffle=True)
            for i, data in enumerate(x_b):
                self.zero_grad_models()
                for k_ind in range(self.k):
                    phi = [self.models[i](data[:,:,i]) for i in range(self.k)]  # each phi[i] should have shape (b,1)
                    e_term = self.calc_exp_term(phi, data)
                    loss = -(sum(phi).mean() - self.eps*e_term + self.eps)
                    loss.backward()
                    self.opt[k_ind].step()
            print(f'finished epoch {epoch}, loss={-loss}')
        self.models_to_eval

    def calc_exp_term(self, phi, x):
        # TD - IMPLEMENT MORE MEMORY EFFICIENT CALCULATION!!! DIVIDE LOSS!!

        # calc loss tensor
        c = self.calc_cost(x)

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
        reshaped_c = sum(reshaped_c)

        return torch.mean(torch.exp((reshaped_term-reshaped_c)/self.eps))


    def calc_cost(self, data):
        """
        calculates the cost over bacthed data
        :param data:
        :return:
        """
        if self.cost == 'quad':
            cost = QuadCost(data)
        elif self.cost == 'quad_gw':
            cost = QuadCostGW(data, self.matrices)
        elif self.cost == 'ip_gw':
            cost = IPCostGW(data, self.matrices)


        # NOW - BROADCAST!!
        return cost

    def zero_grad_models(self):
        for opt in self.opt:
            opt.zero_grad()

    def models_to_eval(self):
        for model in self.models:
            model.eval()


class NE_model(nn.Module):
    def __init__(self, dim, hidden_dim=32):
        super(NE_model, self).__init__()
        self.dim = dim

        # Taos's N-GW model
        self.fc1 = nn.Linear(dim, self.dim)
        self.fc2 = nn.Linear(self.dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x









if __name__ == '__main__':
    main()