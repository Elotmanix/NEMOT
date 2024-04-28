import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from timeit import default_timer as timer
import os
from config import PreprocessMeta


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
    params = PreprocessMeta()

    # params = gen_params()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(params['cuda_visible'])
    device = torch.device("cuda:0" if (torch.cuda.is_available() and params['device'] == 'gpu') else "cpu")
    print(f"Using {device}")

    X = gen_data(params)
    X = X.to(device)

    if params['alg'] == 'ne_mot':
        MOT_agent = MOT_NE_alg(params, device)
        MOT_agent.train_mot(X)





########
def gen_params():
    params = {
        'batch_size': 128,
        'epochs': 2000,
        'lr': 1e-4,
        'n': 50000,
        'k': 2,
        'eps': 0.5,
        'cost': 'quad',  # options - quad, quad_gw, ip_gw
        'alg': 'ne_mot',        #options - ne_mot, sinkhorn_mot,ne_gw, sinkhorn_gw
        'hidden_dim': 32,
        'mod': 'mot',       #options - mot, mgw
        'seed': 42,
        'data_dist': 'uniform',
        # 'dims': [1,1,1,1,1,1,1,1],
        # 'dims': [100,100,100,100,100,100,100,100],
        'dims': [5,5,5,5,5,5],
        'device': 'gpu',
        'cuda_visible': 0
    }
    # TD: ADJUST DIMS TO K
    params['batch_size'] = min(params['batch_size'],params['n'])
    return params


def gen_data(params):
    if params['data_dist'] == 'uniform':
        # generate k samples which are d-dimensional with n samples (from Taos's notebook)
        X = []
        for i in range(params['k']):
            X.append(np.random.uniform(-1/np.sqrt(params['dims'][i]),1/np.sqrt(params['dims'][i]),(params['n'],params['dims'][i])).astype(np.float32))
        X = torch.from_numpy(np.stack(X, axis=-1))

        if params['alg'] not in ('ne_gw','ne_mot'):
            MU = [(1 / params['n']) * np.ones(params['n'])]*params['k']
            return X, MU

        return X


def QuadCost(data, mod='circle'):
    k=data.shape[-1]
    if mod == 'circle':
        differences = [torch.norm(data[:, :, i] - data[:, :, (i + 1) % k], dim=1) for i in range(k)]
    elif mod == 'tree':
        # calculate loss according to tree structure
        pass
    else:
        # calculate all pairwise quadratic losses
        ###
        # option 1 - through broadcasting:
        # Expand 'data' to (n, d, k, k) by repeating it across new dimensions
        data_expanded = data.unsqueeze(3).expand(-1, -1, -1, k)
        data_t_expanded = data.unsqueeze(2).expand(-1, -1, k, -1)

        # Compute differences using broadcasting (resulting shape will be (n, d, k, k))
        differences = data_expanded - data_t_expanded

        # Compute norms (resulting shape will be (n, k, k))
        differences = torch.norm(differences, dim=1)
        ###
        # option 2 - via a nested loop (doesnt use tensor operations but performs half the computations)
        # pairwise_norms = torch.zeros((n, k, k))
        # for i in range(k):
        #     for j in range(i + 1, k):
        #         pairwise_norms[:, i, j] = torch.norm(data[:, :, i] - data[:, :, j], dim=1)
        # differences += pairwise_norms.transpose(1, 2)
        ###


    return differences

class MOT_NE_alg():
    def __init__(self, params, device):
        self.models = []
        self.k = params['k']
        self.num_epochs = params['epochs']
        self.batch_size = params['batch_size']
        self.eps = params['eps']
        for i in range(params['k']):
            model = NE_model(dim=params['dims'][i], hidden_dim=params['hidden_dim'])
            model.to(device)
            self.models.append(model)
        self.cost = params['cost']
        self.opt = [torch.optim.Adam(list(self.models[i].parameters()), lr=params['lr']) for i in range(self.k)]
        self.device = device


    def train_mot(self, X):
        for epoch in range(self.num_epochs):
            x_b = DataLoader(X, batch_size=self.batch_size, shuffle=True)
            l = 0
            t0 = timer()
            for i, data in enumerate(x_b):
                self.zero_grad_models()
                # data.to(self.device)
                for k_ind in range(self.k):
                    phi = [self.models[i](data[:,:,i]) for i in range(self.k)]  # each phi[i] should have shape (b,1)
                    e_term = self.calc_exp_term(phi, data)
                    loss = -(sum(phi).mean() - self.eps*e_term + self.eps)
                    loss.backward()
                    self.opt[k_ind].step()
                l -= loss.item()
            print(f'finished epoch {epoch}, loss={-loss:.5f}, took {timer()-t0:.2f} seconds')
            # print(f'finished epoch {epoch}, loss={l / i}')
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
            cost = QuadCost(data, mod='circle')
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