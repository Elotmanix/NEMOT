import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from timeit import default_timer as timer
import os
from config import PreprocessMeta
from data_fns import gen_data, QuadCost
from models import Net_EOT, NE_mot_model
from TaosBiEOT import dual_loss



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
    print(params)

    X = gen_data(params)
    X = X.to(device)

    # d = params['dims'][0]
    # n = params['n']
    #
    # X = torch.from_numpy(
    #     np.random.uniform(-1 / np.sqrt(d), 1 / np.sqrt(d), (n, d))).float().to(device)  # uniform [-1/sqrt(d),1/sqrt(d)]^d
    # Y = torch.from_numpy(np.random.uniform(-1 / np.sqrt(d), 1 / np.sqrt(d), (n, d))).float().to(device)
    # X = torch.stack([X,Y], dim=-1)

    if params['alg'] == 'ne_mot':
        MOT_agent = MOT_NE_alg(params, device)
        MOT_agent.train_mot(X)





########






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


    def train_mot(self, X):
        x_b = DataLoader(X, batch_size=self.batch_size, shuffle=True)
        tot_loss = []
        times = []
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
            tot_loss.append(-l+ self.eps)
            epoch_time = timer()-t0
            times.append(epoch_time)

            print(f'finished epoch {epoch}, loss={-l+ self.eps:.5f}, took {epoch_time:.2f} seconds')
            # print(f'finished epoch {epoch}, loss={l / i}')
        tot_loss = np.mean(tot_loss[-10:])
        avg_time = np.mean(times)
        print(f'Finished run, loss is {tot_loss:.5f}, average epoch time is {avg_time:.3f} seconds')
        self.models_to_eval

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











if __name__ == '__main__':
    main()